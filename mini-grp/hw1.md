

# HW1: Making a Generalist Robotics Policy

A Generalist Robtoics Policy (GRP) is made up from a modified [vision transformer](https://arxiv.org/abs/2010.11929). A vision transfer is a modified version of a transformer that is designed to process images instead of text. For a transformer to process images the images need to be sliced up into patches that can be tokenized.

You can complete the homework by addressing the #TODO: in the files.

# GRP Transformer Similar to [Octo](https://octo-models.github.io/).

The provided [code](grp_model.py) is an example of a vision transformer. Modifiy this code to become a multi-modal transformer model that accepts images and text as input and outputs either classes or continuous values. Make sure to impliment the block masking to train the model to work when goals are provided via images or text.

### Discrete vs Continuous Action Space [4pts]

Different methods can be used to model the action distribution. Many papers have discretized the action space (e.g. OpenAI Hand), resulting in a good performance. Train the GRP model with a discrete representation with 14 bins for each action dimension (cross-entropy) vs a continuous representation (MSE) and compare the performance of these two distributions. Compare the performance in [simpleEnv](https://github.com/milarobotlearningcourse/SimplerEnv)

**For all the following parts of the assignment, use the continuous representation model for the analysis.**

### Effect of Encoding Size [2pts]

In this section, the goal is to compare training results when using two different encoding sizes: 128 and 256. Remark on the performance difference when using these two different encoding sizes. 

### Replace the Text Encoder with the one from T5 [2pts]

The text tokenization and encoding provided in the initial version of the code are very basic. However, improving the tokenization to use pretrained language embedding tokens may improve goal-text generalization. To this end, use the tokenizer from the [T5 model](https://jmlr.org/papers/v21/20-074.html) to tokenize the text used to encode the goal descriptions. Some example code to get started is available [here](https://huggingface.co/docs/transformers/en/model_doc/t5).

## State History [2pts]

For most robotics problems, a single image is not enough to determine the dynamics well enough to predict the next state. This lack of dynamics information means the model can only solve certain tasks to a limited extent. To provide the model with sufficient state information, update the GRP input to include two images from the state history and evaluate the performance of this new model. Remark on the change in performance.

## Action Chunking [2pts]

One of the many methods used to smooth motion and compensate for multi-modal behaviour in the dataset is to predict many actions at a time. Train two new models that each have chunk size of 4 and 8. Provide plots and remark on the performance of using different chunk sizes.

## Real Hardware Evaluation [2pts]

The last step of training a robotics policy if to test it on real hardware. This is the only relaible option to evalaute performance. Run you policy on the real hardware and record a video of the behaviour to include in your submission. See the [documentation](https://docs.google.com/document/d/1CdADPr5R9lAAyaZEJ6uYGImnCk5vshCwAWYD9ofBV24/edit?usp=sharing) for details on how to run your policy on the hardware.


## Tips:

1. If you are having trouble training the model and not running out of memory, use a smaller batch size and gradient accumulation. Training will take a little longer but it should work.

# Submitting the code and experiment runs

In order to turn in your code and experiment logs, create a folder that
contains the following:

-   A folder named `data` with all the experiment runs from this assignment. **Do not change the names originally assigned to the folders, as specified by `exp_name` in the instructions. Video logging is not utilized in this assignment, as visualizations are provided through plots, which are outputted during training.**

-   The `roble` folder with all the `.py` files, with the same names and directory structure as the original homework repository (excluding the `data` folder). Also include any special instructions we need to run in order to produce each of your figures or tables (e.g. "run python myassignment.py -sec2q1" to generate the result for Section 2 Question 1) in the form of a README file.

As an example, the unzipped version of your submission should result in the following file structure. **Make sure that the submit.zip file is below 15MB and that they include the prefix `mini-grp/...`, `data/q1_`, `data/q2_`, `data/q3_`, etc.**


If you are a Mac user, **do not use the default "Compress" option to create the zip**. It creates artifacts that the autograder does not like. You may use `zip -vr submit.zip submit -x "*.DS_Store"` from your terminal.

Turn in your assignment on Gradescope. Upload the zip file with your code and log files to **HW Code**, and upload the PDF of your report to **HW**.

## Submitting Model to Leader Board

There is a [leaderboard](https://huggingface.co/spaces/gberseth/mila-robot-learning-course) that can be used to evalaute model. If you don't have LIBERO working or SimpleEnv you can submit your trained policies to this page for evalaution. Also, everyone must submit their model to this leaderboard for autonomous evalaution. 


Submit your model by uploading the files in your output folder for the model you want evaluated. 
```
/playground/mini-grp/outputs/2026-01-28/10-24-45$ hf upload gberseth/mini-grp .
```

Make sure these files exist in this folder
```
-rw-rw-r-- 1 gberseth gberseth   14251 Jan 28 10:25 grp_model.py
drwxrwxr-x 2 gberseth gberseth    4096 Jan 28 10:24 .hydra
-rw-rw-r-- 1 gberseth gberseth 1112018 Jan 28 10:25 miniGRP.pth
```

The evalaution code will look for your .pth file, which needs you grp_model.py file to understand how to use your model, and the .hydra folder to load your model configuration.
