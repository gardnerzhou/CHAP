python train_DS_dist_2D.py  --gpu 0 --projectdim 16 --subexp run1 && \
python train_final_3D.py  --gpu 2 --w_f 0.5 --exp lamda_0.5 && \
python train_final_3D.py  --gpu 2 --w_f 1.5 --exp lamda_1.5 && \
python train_final_3D.py  --gpu 2 --w_f 3.0 --exp lamda_3.0 




