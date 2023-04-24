#  Final Project: Testing Adversarial Attacks

### Your name and any other group members, if any.

Bennett Brain and Chandler Smith

### Project Description

The primary objective of this project is to investigate the impact of different adversarial attacks on the performance of machine learning models. We aim to analyze the effects of various attack types on model performance and vulnerability. By comparing white and black box attack types, we hope to gain valuable insights into how the model's performance is influenced by different adversaries.

### Links/URLs to any videos you created and want to submit as part of your report.
Final Presentation: https://drive.google.com/file/d/1s-lNFUab-5qC6RyE2-qvSIVKNiljwggH/view?usp=sharing

Final Presentation Slides: https://docs.google.com/presentation/d/1MynTl37XqaN62oFo0aP-VAFZtl9mfZrFQ6JzQAzTTuw/edit?usp=sharing

Final Paper: https://github.com/chansmi/CV_Adversarial_Examples/blob/master/FinalPaperFormatted.pdf


### What operating system and IDE you used to run and compile your code.
With minor tweaks, this system should run on windows or mac, however, the uploaded code is optimized for a mac using VSCode. For testing purposes, using VSCode would be the preferred method. 


### Instructions for running your executables.
As long as the appropriate files are included, this code should run mostly untouched. Prior to running the code, however, there are two places where you will need to change the absolute path of the file so it saves correctly. Depending on if you if you want to run a PGD or a FGSM attack, comment out line 225 on white_box.py.
For the black box attacks: In blackbox.py, comment in lines 54-60 to use targeted attacks, and comment out lines 63-69 to remove random attacks.  Each specific line that begins with at.[name] is a type of attack- comment in the attack you would like to try. On line 132. the specific digits to target can be changed.  Any list of digits in the range 0-9 should work. Lines 164-169 should mirror lines 54-60 to ensure the visualizations match the attacks being applied.


