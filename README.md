# Thesis-Project
This project trains an AI model on spoken speech and environmental sound that is deployed on a Raspberry Pi 5 to denoise audio. This project also includes the script used on the Raspberry Pi to complete this implementation.

### Relevant Steps ###
1. Store the data you want to train the model on into Google Drive or adjust the code to download it from the proper source.
2. Run the Google Colab model. (you might want to use the TPU runtime to ensure you have enough RAM to process the data. If the RAM runs out, the runtime will crash and you will have to start over).
3. (optional) Test the model by running the prediction function on an audio file and observe the HASPI and SNR values and listen to the processed audio file.
4. Download the tflite file for the compressed model.
5. Set up the Raspeberry Pi 5 or other hardware. For this implementation, I used sudo apt-get upgrade and then set up the configuration to enable SSH, I2C, and Serial). Then, install the required packages: scipy, numpy, portaudio19-dev, python3-pyaudio, python3-matplotlib, python3-pandas, python3-tflite_runtime, and cv2. NOTE: you might have to break the system package system in order for the packages to install correctly. I input exactly: $python3 -m pip install tflite-runtime --break-system-packages for this to work.
6. Set up your audio input source through the Bluetooth menu or manaully using the audio_test.py script.
7. Store your tflite model, audio files you want to test the model with, and the audio_test.py script in the same directory (mine was /home/myusername)
8. Run the script using something like $python3 audio_test.py --ai 1. If this doesn't work, check that the definitive paths and files you are trying to access are all in the same location and the paths are correct!
   
### Acknowledgements ###
The code was developed to complete a Masters degree at the University of St. Thomas. Thank you to all of the people involved in making this a reality!

### References ###
Additionally, inspiration for this project was derived from the following sources:
* https://medium.com/@vaibhavtalekar87/a-deep-dive-into-audio-denoising-with-tensorflow-cnn-a996e0c62e16
* https://github.com/karolpiczak/ESC-50
* https://github.com/rdadlaney/Audio-Denoiser-CNN
* https://github.com/mkvenkit/simple_audio_pi/blob/main/audio_test.py
* https://pidora.ca/transform-your-raspberry-pi-into-a-powerful-signal-processing-workstation/

And these following sources for other implementation needs
* https://librosa.org/doc/latest/generated/librosa.get_duration.html
* https://claritychallenge.org/clarity/_modules/clarity/evaluator/haspi/haspi.html
* https://colab.research.google.com/github/pytorch/audio/blob/gh-pages/main/_downloads/68fedf50687e692876b68727022ad06e/audio_resampling_tutorial.ipynb#scrollTo=vZ4lI_HYN9kO
