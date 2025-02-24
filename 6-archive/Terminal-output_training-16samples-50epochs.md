Terminal output from training 

DatasetDict({
    train: Dataset({
        features: ['text', 'label', 'label_text'],
        num_rows: 67
    })
    test: Dataset({
        features: ['text', 'label', 'label_text'],
        num_rows: 67
    })
})

 === Training data ===
Counter({'narrator': 44, 'character': 23})

 === Test data ===
Counter({'narrator': 47, 'character': 20})
model_head.pkl not found on HuggingFace Hub, initialising classification head with random weights. You should TRAIN this model on a downstream task to use it for predictions and inference.
/home/christof/.local/lib/python3.10/site-packages/setfit/data.py:154: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
  df = df.apply(lambda x: x.sample(min(num_samples, len(x)), random_state=seed))
Map: 100%|██████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 8520.68 examples/s]
***** Running training *****
  Num unique pairs = 544
  Batch size = 8
  Num epochs = 50
  Total optimization steps = 3400
  0%|                                                                                           | 0/3400 [00:00<?, ?it/s]
{'embedding_loss': 0.2744, 'learning_rate': 5.882352941176471e-08, 'epoch': 0.01}               | 0/3400 [00:00<?, ?it/s]
{'embedding_loss': 0.2109, 'learning_rate': 2.9411764705882355e-06, 'epoch': 0.74}                                       
{'embedding_loss': 0.0059, 'learning_rate': 5.882352941176471e-06, 'epoch': 1.47}                                        
{'embedding_loss': 0.0004, 'learning_rate': 8.823529411764707e-06, 'epoch': 2.21}                                        
{'embedding_loss': 0.0002, 'learning_rate': 1.1764705882352942e-05, 'epoch': 2.94}                                       
{'embedding_loss': 0.0003, 'learning_rate': 1.4705882352941179e-05, 'epoch': 3.68}                                       
{'embedding_loss': 0.0002, 'learning_rate': 1.7647058823529414e-05, 'epoch': 4.41}                                       
{'embedding_loss': 0.0001, 'learning_rate': 1.993464052287582e-05, 'epoch': 5.15}                                        
{'embedding_loss': 0.0001, 'learning_rate': 1.9607843137254903e-05, 'epoch': 5.88}                                       
{'embedding_loss': 0.0002, 'learning_rate': 1.9281045751633988e-05, 'epoch': 6.62}                                       
{'embedding_loss': 0.0001, 'learning_rate': 1.8954248366013074e-05, 'epoch': 7.35}                                       
{'embedding_loss': 0.0001, 'learning_rate': 1.862745098039216e-05, 'epoch': 8.09}                                        
{'embedding_loss': 0.0001, 'learning_rate': 1.8300653594771242e-05, 'epoch': 8.82}                                       
{'embedding_loss': 0.0001, 'learning_rate': 1.7973856209150328e-05, 'epoch': 9.56}                                       
{'embedding_loss': 0.0001, 'learning_rate': 1.7647058823529414e-05, 'epoch': 10.29}                                      
{'embedding_loss': 0.0001, 'learning_rate': 1.73202614379085e-05, 'epoch': 11.03}                                        
{'embedding_loss': 0.0, 'learning_rate': 1.6993464052287582e-05, 'epoch': 11.76}                                         
{'embedding_loss': 0.0001, 'learning_rate': 1.6666666666666667e-05, 'epoch': 12.5}                                       
{'embedding_loss': 0.0, 'learning_rate': 1.6339869281045753e-05, 'epoch': 13.24}                                         
{'embedding_loss': 0.0001, 'learning_rate': 1.601307189542484e-05, 'epoch': 13.97}                                       
{'embedding_loss': 0.0, 'learning_rate': 1.568627450980392e-05, 'epoch': 14.71}                                          
{'embedding_loss': 0.0001, 'learning_rate': 1.5359477124183007e-05, 'epoch': 15.44}                                      
{'embedding_loss': 0.0001, 'learning_rate': 1.5032679738562093e-05, 'epoch': 16.18}                                      
{'embedding_loss': 0.0, 'learning_rate': 1.4705882352941179e-05, 'epoch': 16.91}                                         
{'embedding_loss': 0.0001, 'learning_rate': 1.4379084967320263e-05, 'epoch': 17.65}                                      
{'embedding_loss': 0.0, 'learning_rate': 1.4052287581699347e-05, 'epoch': 18.38}                                         
{'embedding_loss': 0.0001, 'learning_rate': 1.3725490196078432e-05, 'epoch': 19.12}                                      
{'embedding_loss': 0.0, 'learning_rate': 1.3398692810457516e-05, 'epoch': 19.85}                                         
{'embedding_loss': 0.0001, 'learning_rate': 1.3071895424836602e-05, 'epoch': 20.59}                                      
{'embedding_loss': 0.0, 'learning_rate': 1.2745098039215686e-05, 'epoch': 21.32}                                         
{'embedding_loss': 0.0001, 'learning_rate': 1.2418300653594772e-05, 'epoch': 22.06}                                      
{'embedding_loss': 0.0001, 'learning_rate': 1.2091503267973856e-05, 'epoch': 22.79}                                      
{'embedding_loss': 0.0, 'learning_rate': 1.1764705882352942e-05, 'epoch': 23.53}                                         
{'embedding_loss': 0.0, 'learning_rate': 1.1437908496732026e-05, 'epoch': 24.26}                                         
{'embedding_loss': 0.0, 'learning_rate': 1.1111111111111113e-05, 'epoch': 25.0}                                          
{'embedding_loss': 0.0, 'learning_rate': 1.0784313725490196e-05, 'epoch': 25.74}                                         
{'embedding_loss': 0.0, 'learning_rate': 1.0457516339869283e-05, 'epoch': 26.47}                                         
{'embedding_loss': 0.0, 'learning_rate': 1.0130718954248367e-05, 'epoch': 27.21}                                         
{'embedding_loss': 0.0, 'learning_rate': 9.803921568627451e-06, 'epoch': 27.94}                                          
{'embedding_loss': 0.0, 'learning_rate': 9.477124183006537e-06, 'epoch': 28.68}                                          
{'embedding_loss': 0.0001, 'learning_rate': 9.150326797385621e-06, 'epoch': 29.41}                                       
{'embedding_loss': 0.0, 'learning_rate': 8.823529411764707e-06, 'epoch': 30.15}                                          
{'embedding_loss': 0.0, 'learning_rate': 8.496732026143791e-06, 'epoch': 30.88}                                          
{'embedding_loss': 0.0, 'learning_rate': 8.169934640522877e-06, 'epoch': 31.62}                                          
{'embedding_loss': 0.0, 'learning_rate': 7.84313725490196e-06, 'epoch': 32.35}                                           
{'embedding_loss': 0.0, 'learning_rate': 7.516339869281046e-06, 'epoch': 33.09}                                          
{'embedding_loss': 0.0, 'learning_rate': 7.189542483660131e-06, 'epoch': 33.82}                                          
{'embedding_loss': 0.0, 'learning_rate': 6.862745098039216e-06, 'epoch': 34.56}                                          
{'embedding_loss': 0.0, 'learning_rate': 6.535947712418301e-06, 'epoch': 35.29}                                          
{'embedding_loss': 0.0, 'learning_rate': 6.209150326797386e-06, 'epoch': 36.03}                                          
{'embedding_loss': 0.0, 'learning_rate': 5.882352941176471e-06, 'epoch': 36.76}                                          
{'embedding_loss': 0.0, 'learning_rate': 5.555555555555557e-06, 'epoch': 37.5}                                           
{'embedding_loss': 0.0, 'learning_rate': 5.2287581699346416e-06, 'epoch': 38.24}                                         
{'embedding_loss': 0.0, 'learning_rate': 4.901960784313726e-06, 'epoch': 38.97}                                          
{'embedding_loss': 0.0, 'learning_rate': 4.5751633986928105e-06, 'epoch': 39.71}                                         
{'embedding_loss': 0.0, 'learning_rate': 4.2483660130718954e-06, 'epoch': 40.44}                                         
{'embedding_loss': 0.0, 'learning_rate': 3.92156862745098e-06, 'epoch': 41.18}                                           
{'embedding_loss': 0.0, 'learning_rate': 3.5947712418300657e-06, 'epoch': 41.91}                                         
{'embedding_loss': 0.0, 'learning_rate': 3.2679738562091506e-06, 'epoch': 42.65}                                         
{'embedding_loss': 0.0, 'learning_rate': 2.9411764705882355e-06, 'epoch': 43.38}                                         
{'embedding_loss': 0.0, 'learning_rate': 2.6143790849673208e-06, 'epoch': 44.12}                                         
{'embedding_loss': 0.0, 'learning_rate': 2.2875816993464053e-06, 'epoch': 44.85}                                         
{'embedding_loss': 0.0, 'learning_rate': 1.96078431372549e-06, 'epoch': 45.59}                                           
{'embedding_loss': 0.0, 'learning_rate': 1.6339869281045753e-06, 'epoch': 46.32}                                         
{'embedding_loss': 0.0, 'learning_rate': 1.3071895424836604e-06, 'epoch': 47.06}                                         
{'embedding_loss': 0.0, 'learning_rate': 9.80392156862745e-07, 'epoch': 47.79}                                           
{'embedding_loss': 0.0, 'learning_rate': 6.535947712418302e-07, 'epoch': 48.53}                                          
{'embedding_loss': 0.0, 'learning_rate': 3.267973856209151e-07, 'epoch': 49.26}                                          
{'embedding_loss': 0.0, 'learning_rate': 0.0, 'epoch': 50.0}                                                             
{'train_runtime': 1281.6253, 'train_samples_per_second': 21.223, 'train_steps_per_second': 2.653, 'epoch': 50.0}         
100%|████████████████████████████████████████████████████████████████████████████████| 3400/3400 [21:21<00:00,  2.65it/s]
***** Running evaluation *****███████████████████████████████████████████████████████| 3400/3400 [21:21<00:00,  2.78it/s]

=== Evaluation ===
 {'accuracy': 0.8955223880597015}
christof@aorus:/data/Seafile/01_DH-Professur/02_Lehre/2024-Sommer/Advanced-Topics/SetFit/french$