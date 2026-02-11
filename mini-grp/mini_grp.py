from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
import dill
import hydra
from omegaconf import DictConfig, OmegaConf
import threading
from queue import Queue
import torch
import numpy as np

def get_inverse_sqrt_lambda(optimizer, warmup_steps):
    """
    Creates a lambda function for an inverse square root learning rate schedule 
    with a linear warmup phase.
    """
    def lr_lambda(current_step):
        # Ensure the step is at least 1 to avoid division by zero or log issues
        current_step += 1 
        # Linear warmup phase
        if current_step < warmup_steps:
            return float(current_step) / float(warmup_steps)
        # Inverse square root decay phase
        else:
            return float(warmup_steps) / float(current_step)**0.5
    return lr_lambda

@hydra.main(config_path="./conf", config_name="64pix-pose")
def my_main(cfg: DictConfig):
    torch.manual_seed(cfg.r_seed)
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print ("cfg:", OmegaConf.to_yaml(cfg))
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    # cfg.device = device

    wandb = None
    if not cfg.testing:
        import wandb
        # start a new wandb run to track this script
        wandb.init(
            project=cfg.experiment.project,
            # track hyperparameters and run metadata
            config= OmegaConf.to_container(cfg),
            name=cfg.experiment.name,
        )
        wandb.run.log_code(".")

    tokenizer = None
    text_model = None
    if cfg.dataset.encode_with_t5: ## Load T5 model
        # TODO:    
        # Load the t5 model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        text_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        text_model.eval()
        text_model.to(cfg.device)

    from mini_shuffel_buffer import CircularBuffer, get_dataset_portion

    # Load model based on configuration
    if cfg.model.type == "convnet":
        from grp_convnet import GRPConvNet
        from grp_convnet import estimate_loss
        model = GRPConvNet(cfg)
        print(f"Using ConvNet model")
    if cfg.model.type == "dense":
        from grp_convnet import PoseOnlyNet
        from grp_convnet import estimate_loss
        model = PoseOnlyNet(cfg)
        print(f"Using Dense model")
    else:
        from grp_model import GRP
        from grp_model import estimate_loss
        model = GRP(cfg)
        print(f"Using Transformer model")
    
    from sim_eval import eval_model_in_sim
    model.to(cfg.device)
    cBuffer = CircularBuffer(cfg.dataset.buffer_size, cfg, model)
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    ## Print the amount of memory used by the model
    print("Memory used by the model:", torch.cuda.memory_allocated(cfg.device) / 1e6, "MB")
    ## Print the amount of memory used by the dataset cBuffer
    from pympler import asizeof
    cBuffer.print_mem_footprint()

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.1)
    # Select learning rate schedule based on configuration
    schedule_type = getattr(cfg, 'lr_schedule', 'inverse_sqrt')  # default to inverse_sqrt
    if schedule_type == 'linear':
        import torch.optim.lr_scheduler as lr_scheduler
        from torch.optim.lr_scheduler import LinearLR
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=cfg.max_iters)
    else:  # inverse_sqrt
        lr_lambda = get_inverse_sqrt_lambda(optimizer, warmup_steps=1000)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    if "simpler_env" in cfg.simEval:
        import simpler_env
        task_name = "widowx_carrot_on_plate"  # @param ["google_robot_pick_coke_can", "google_robot_move_near", "google_robot_open_drawer", "google_robot_close_drawer", "widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube", "widowx_put_eggplant_in_basket"]
        if 'env' in locals():
            print("Closing existing env")
            env.close()
            del env
        env = simpler_env.make(task_name)
        env_unwrapped = env.env.env.env ## Updated gymnasium wrapper adds lots of wrappers.

    shared_queue = Queue(maxsize=1)
    data_thread = threading.Thread(target=cBuffer.shuffle, args=(shared_queue,))
    data_thread.start()

    # Initialize cProfile if enabled
    profiler = None
    if cfg.profiler.enable:
        import cProfile, pstats
        profiler = cProfile.Profile()
        profiler.enable()

    for iter in range(cfg.max_iters+1):
        if iter % cfg.eval_interval == 0 or iter == cfg.max_iters - 1:
            losses = estimate_loss(model, cBuffer)
            print(f"step {iter}: train loss {losses['train']:.6f}, val loss {losses['val']:.6f}, memory {torch.cuda.memory_allocated(cfg.device) / 1e6:.2f} MB")
            if not cfg.testing:
                wandb.log({"train loss": losses['train'], "val loss": losses['val'],
                        "memory": torch.cuda.memory_allocated(cfg.device) / 1e6,
                        "buffer_size": asizeof.asizeof(cBuffer) / 1e6}, step=iter)
            if cfg.profiler.enable:
                stats = pstats.Stats(profiler)
                stats.sort_stats(pstats.SortKey.TIME).print_stats(20)

        if iter % cfg.data_shuffel_interval == 0 or iter == cfg.max_iters - 1:
            path_ = "./miniGRP.pth"
            torch.save(model, path_, pickle_module=dill) ## serialize class objects as well.
            ## Save the grp_model.py file into the output folder as well
            import shutil
            shutil.copy(hydra.utils.get_original_cwd()+"/mini-grp/grp_model.py", log_dir)
            print("Model saved to " + path_)
        
        if cfg.simEval and (iter % cfg.eval_vid_iters == 0) and (iter !=0): ## Do this eval infrequently because it takes a fiar bit of compute
            if "simpler_env" in cfg.simEval:
                # Note: moved import of `eval_model_in_sim` into `my_main` to avoid circular imports
                eval_model_in_sim(cfg, model, cfg.device, log_dir, env, env_unwrapped, 
                            wandb=wandb, iter_=iter, tokenizer=tokenizer, text_model=text_model)
            if "libero" in cfg.simEval:
                from sim_eval import eval_libero
                eval_libero(model, device=cfg.device, cfg=cfg, iter_=iter, log_dir=log_dir, 
                            tokenizer=tokenizer, text_model=text_model, wandb=wandb)

        if iter % cfg.data_shuffel_interval == 0 and iter > 0:
            ## Update the dataset
            shared_queue.put('shuffle')

        xb, xp, xg, xgi, yb, last_action = cBuffer.get_batch_grp('train', cfg, cfg.batch_size)

        # evaluate the loss
        # Block masking: train with either text OR image goals (randomly) if enabled.
        mask_ = None
        if cfg.policy.random_masking_enabled:
            mask_ = (np.random.rand() < 0.5)  # True -> mask goal image tokens (use text); False -> mask text (use image)
        logits, loss = model(xb, xg, xgi, yb, pose=xp, last_action=last_action, mask_=mask_)
        
        # backward pass
        loss.backward()
        # max_norm is the maximum allowed norm of the gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if (iter + 1) % cfg.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    if not cfg.testing:
        wandb.finish()
    shared_queue.put(None)
    data_thread.join()
    
    # Disable and print profiling results if enabled
    if profiler is not None:
        profiler.disable()
        import pstats
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        print("\n=== cProfile Results ===")
        stats.print_stats(30)  # Print top 30 functions
        stats.dump_stats("profile.prof")
        print("Profile saved to profile.prof")
    
    return losses['val']
 
if __name__ == "__main__":
    results = my_main()
    print("results:", results)