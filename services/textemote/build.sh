echo Training the spaCy project...
cd textcat_goemotions
echo Limit training time by changing a few hyperparameters...
sed -i 's/max_steps = 20000/max_steps = 20000/' "configs/cnn.cfg"
spacy project run all
cd ..
echo Training complete. Model is available at textcat_goemotions/training/cnn/model-best
