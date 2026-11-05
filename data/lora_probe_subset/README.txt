Two-clip subset used ONLY for the training memory and timing probe.
It exists because the VAE pre-encode is O(number of clips) and is paid once per job;
measuring seconds-per-step does not need the whole set, and job 50226994 hit its wall
clock before reaching the step loop with all 16 clips. Not used for real training.
