mode=$1
dir_for_saving=$2
model_name=$3
path_to_save_physion_dataset=$4
gpu=$5

# Print the parameters for logging or confirmation
echo "Mode: $mode"
echo "Directory for Saving: $dir_for_saving"
echo "Model Name: $model_name"
echo "Path to Physion Dataset: $path_to_save_physion_dataset"

# Run the shell command, passing the Python variables using f-strings
physion_feature_extract \
--data_root_path $path_to_save_physion_dataset \
--model_class ${model_name} \
--gpu $gpu \
--batch_size 8 \
--dir_for_saving ${dir_for_saving} \
--mode ${mode}

physion_train_readout \
--train-path ${dir_for_saving}/${mode}/train_features.hdf5 \
--test-path ${dir_for_saving}/${mode}/test_features.hdf5 \
--model-name ${model_name} \
--train-scenario-indices ${dir_for_saving}/${mode}/train_json.json \
--test-scenario-indices ${dir_for_saving}/${mode}/test_json.json \
--test-scenario-map ${dir_for_saving}/${mode}/test_scenario_map.json \
--save_path ${dir_for_saving} \
--clf_C 1e-6 1e-5 0.01 0.1 1 5 10 20
