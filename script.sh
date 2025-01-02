
#To train the diffusion model, adjust the --conditioning_key argument if you want to experiment with different types of control:
python DiffusionVel.py  --if_train --batch_size=16 --learning_rate=1e-4  --check_val_every_n_epoch=10 --anno_path=./split_files --train_anno=curvefault_b_train_test.txt --val_anno=curvefault_b_test_test.txt --test_anno=curvefault_b_test_test.txt --dataset=curvefault-b  --max_epochs=200 --model_config=./models/dpm.yaml --conditioning_key=geo_concat 

#To test the trained GDM, run:
python DiffusionVel.py   --batch_size=16 --anno_path=./split_files --test_anno=curvefault_b_test_test.txt --dataset=curvefault-b --model_config=./models/dpm.yaml --conditioning_key=geo_concat --geo=./lightning_logs/flatfault_b_unconditional/checkpoints/epoch=99-step=48999.ckpt 

#To test the multi-information integration with seismic, well, background, and geological data, adjust the arguments for the factors and input paths:
python DiffusionVel.py --factor_0=0.25 --factor_1=0.5 --factor_2=0.25 --batch_size=16 --anno_path=./split_files --test_anno=curvefault_b_test_test.txt --dataset=curvefault-b --model_config=./models/dpm.yaml --conditioning_key=seis_well_back_concat --well=your_path_well --back=your_path_back --seis=your_path_seis --geo=your_path_geo