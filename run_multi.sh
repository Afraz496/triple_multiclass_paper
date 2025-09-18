output_dir="manuscript_updates"
data_dir="data/triple_3viruses_nn_data.csv"

#python train_test.py -o $output_dir -d $data_dir
python rerun_ml_pipeline.py -o $output_dir -d $data_dir --use_saved_processor "Multiclass/processor.pkl" --use_saved_model "Multiclass/models/nn_model3.h5"