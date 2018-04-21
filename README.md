This submission is broken down into two different sections. The first section provided within this zip file provides the annotations used in demonstrations, along with the primary source code.
If visualisations are all that are required, the program can be run from main.py found at the root of this zip file.



If it is required to generate more data, however, there are unfortunately more steps required. Given the size limit for file submissionss, it is not feasible to supply the RetinaNet model that is trained for our data.
Instead, we provide a link to a GitHub repository where a full version this can be downloaded.  Please note, however, that this model also contains some dependencies on a specific version of the Keras-RetinaNet repository. Instructions for obtaining this are also found below.

Firstly, the library must be installed in order for the model to be loaded.  This can be found at: https://github.com/fizyr/keras-retinanet 

However, at the time of our model's generation, the repository was a lot less mature, hence our model not being compatible with the current version. To load our model, please checkout commit '284228715002590b330184ad8aac9519392d4a7b', which carries an additional dependency on Keras-Resnet, found here: https://github.com/raghakot/keras-resnet

The overall model, as well as the source code seen here, can then be found in the following GitHub repository: https://github.com/Willven/final-project

With these requirements met, footage annotations can be generated using the gen_data.py script with the following usage:
 - python gen_data.py <video_file> <output_file.h5>
