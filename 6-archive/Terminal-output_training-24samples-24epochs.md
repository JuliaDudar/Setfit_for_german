DatasetDict({
    train: Dataset({
        features: ['text', 'label', 'label_text'],
        num_rows: 101
    })
    test: Dataset({
        features: ['text', 'label', 'label_text'],
        num_rows: 101
    })
})

 === Training data ===
Counter({'narrator': 67, 'character': 34})

 === Test data ===
Counter({'narrator': 72, 'character': 29})
model_head.pkl not found on HuggingFace Hub, initialising classification head with random weights. You should TRAIN this model on a downstream task to use it for predictions and inference.
/home/christof/.local/lib/python3.10/site-packages/setfit/data.py:154: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
  df = df.apply(lambda x: x.sample(min(num_samples, len(x)), random_state=seed))
Map: 100%|█████████████████████████████████████████████████████████████████████| 48/48 [00:00<00:00, 13030.00 examples/s]
***** Running training *****
  Num unique pairs = 1200
  Batch size = 8
  Num epochs = 24
  Total optimization steps = 3600
  0%|                                                                                           | 0/3600 [00:00<?, ?it/s]
{'embedding_loss': 0.429, 'learning_rate': 5.555555555555556e-08, 'epoch': 0.01}                | 0/3600 [00:00<?, ?it/s]
{'embedding_loss': 0.2464, 'learning_rate': 2.7777777777777783e-06, 'epoch': 0.33}                                       
{'embedding_loss': 0.0212, 'learning_rate': 5.555555555555557e-06, 'epoch': 0.67}                                        
{'embedding_loss': 0.001, 'learning_rate': 8.333333333333334e-06, 'epoch': 1.0}                                          
{'embedding_loss': 0.0002, 'learning_rate': 1.1111111111111113e-05, 'epoch': 1.33}                                       
{'embedding_loss': 0.0003, 'learning_rate': 1.388888888888889e-05, 'epoch': 1.67}                                        
{'embedding_loss': 0.0001, 'learning_rate': 1.6666666666666667e-05, 'epoch': 2.0}                                        
{'embedding_loss': 0.0002, 'learning_rate': 1.9444444444444445e-05, 'epoch': 2.33}                                       
{'embedding_loss': 0.0002, 'learning_rate': 1.9753086419753087e-05, 'epoch': 2.67}                                       
{'embedding_loss': 0.0001, 'learning_rate': 1.9444444444444445e-05, 'epoch': 3.0}                                        
{'embedding_loss': 0.0001, 'learning_rate': 1.9135802469135804e-05, 'epoch': 3.33}                                       
{'embedding_loss': 0.0001, 'learning_rate': 1.8827160493827163e-05, 'epoch': 3.67}                                       
{'embedding_loss': 0.0001, 'learning_rate': 1.851851851851852e-05, 'epoch': 4.0}                                         
{'embedding_loss': 0.0001, 'learning_rate': 1.820987654320988e-05, 'epoch': 4.33}                                        
{'embedding_loss': 0.0001, 'learning_rate': 1.7901234567901236e-05, 'epoch': 4.67}                                       
{'embedding_loss': 0.0001, 'learning_rate': 1.7592592592592595e-05, 'epoch': 5.0}                                        
{'embedding_loss': 0.0, 'learning_rate': 1.728395061728395e-05, 'epoch': 5.33}                                           
{'embedding_loss': 0.0001, 'learning_rate': 1.697530864197531e-05, 'epoch': 5.67}                                        
{'embedding_loss': 0.0, 'learning_rate': 1.6666666666666667e-05, 'epoch': 6.0}                                           
{'embedding_loss': 0.0001, 'learning_rate': 1.6358024691358026e-05, 'epoch': 6.33}                                       
{'embedding_loss': 0.0001, 'learning_rate': 1.6049382716049385e-05, 'epoch': 6.67}                                       
{'embedding_loss': 0.0, 'learning_rate': 1.5740740740740744e-05, 'epoch': 7.0}                                           
{'embedding_loss': 0.0, 'learning_rate': 1.54320987654321e-05, 'epoch': 7.33}                                            
{'embedding_loss': 0.0001, 'learning_rate': 1.5123456790123458e-05, 'epoch': 7.67}                                       
{'embedding_loss': 0.0, 'learning_rate': 1.4814814814814815e-05, 'epoch': 8.0}                                           
{'embedding_loss': 0.0, 'learning_rate': 1.4506172839506174e-05, 'epoch': 8.33}                                          
{'embedding_loss': 0.0001, 'learning_rate': 1.4197530864197532e-05, 'epoch': 8.67}                                       
{'embedding_loss': 0.0, 'learning_rate': 1.388888888888889e-05, 'epoch': 9.0}                                            
{'embedding_loss': 0.0, 'learning_rate': 1.3580246913580248e-05, 'epoch': 9.33}                                          
{'embedding_loss': 0.0001, 'learning_rate': 1.3271604938271605e-05, 'epoch': 9.67}                                       
{'embedding_loss': 0.0, 'learning_rate': 1.2962962962962964e-05, 'epoch': 10.0}                                          
{'embedding_loss': 0.0, 'learning_rate': 1.2654320987654323e-05, 'epoch': 10.33}                                         
{'embedding_loss': 0.0001, 'learning_rate': 1.234567901234568e-05, 'epoch': 10.67}                                       
{'embedding_loss': 0.0, 'learning_rate': 1.2037037037037039e-05, 'epoch': 11.0}                                          
{'embedding_loss': 0.0, 'learning_rate': 1.1728395061728398e-05, 'epoch': 11.33}                                         
{'embedding_loss': 0.0001, 'learning_rate': 1.1419753086419753e-05, 'epoch': 11.67}                                      
{'embedding_loss': 0.0001, 'learning_rate': 1.1111111111111113e-05, 'epoch': 12.0}                                       
{'embedding_loss': 0.0, 'learning_rate': 1.0802469135802469e-05, 'epoch': 12.33}                                         
{'embedding_loss': 0.0, 'learning_rate': 1.0493827160493827e-05, 'epoch': 12.67}                                         
{'embedding_loss': 0.0, 'learning_rate': 1.0185185185185186e-05, 'epoch': 13.0}                                          
{'embedding_loss': 0.0, 'learning_rate': 9.876543209876543e-06, 'epoch': 13.33}                                          
{'embedding_loss': 0.0, 'learning_rate': 9.567901234567902e-06, 'epoch': 13.67}                                          
{'embedding_loss': 0.0, 'learning_rate': 9.25925925925926e-06, 'epoch': 14.0}                                            
{'embedding_loss': 0.0, 'learning_rate': 8.950617283950618e-06, 'epoch': 14.33}                                          
{'embedding_loss': 0.0, 'learning_rate': 8.641975308641975e-06, 'epoch': 14.67}                                          
{'embedding_loss': 0.0001, 'learning_rate': 8.333333333333334e-06, 'epoch': 15.0}                                        
{'embedding_loss': 0.0, 'learning_rate': 8.024691358024692e-06, 'epoch': 15.33}                                          
{'embedding_loss': 0.0, 'learning_rate': 7.71604938271605e-06, 'epoch': 15.67}                                           
{'embedding_loss': 0.0, 'learning_rate': 7.4074074074074075e-06, 'epoch': 16.0}                                          
{'embedding_loss': 0.0, 'learning_rate': 7.098765432098766e-06, 'epoch': 16.33}                                          
{'embedding_loss': 0.0001, 'learning_rate': 6.790123456790124e-06, 'epoch': 16.67}                                       
{'embedding_loss': 0.0, 'learning_rate': 6.481481481481482e-06, 'epoch': 17.0}                                           
{'embedding_loss': 0.0, 'learning_rate': 6.17283950617284e-06, 'epoch': 17.33}                                           
{'embedding_loss': 0.0, 'learning_rate': 5.864197530864199e-06, 'epoch': 17.67}                                          
{'embedding_loss': 0.0, 'learning_rate': 5.555555555555557e-06, 'epoch': 18.0}                                           
{'embedding_loss': 0.0, 'learning_rate': 5.246913580246914e-06, 'epoch': 18.33}                                          
{'embedding_loss': 0.0, 'learning_rate': 4.938271604938272e-06, 'epoch': 18.67}                                          
{'embedding_loss': 0.0, 'learning_rate': 4.62962962962963e-06, 'epoch': 19.0}                                            
{'embedding_loss': 0.0, 'learning_rate': 4.3209876543209875e-06, 'epoch': 19.33}                                         
{'embedding_loss': 0.0, 'learning_rate': 4.012345679012346e-06, 'epoch': 19.67}                                          
{'embedding_loss': 0.0, 'learning_rate': 3.7037037037037037e-06, 'epoch': 20.0}                                          
{'embedding_loss': 0.0, 'learning_rate': 3.395061728395062e-06, 'epoch': 20.33}                                          
{'embedding_loss': 0.0, 'learning_rate': 3.08641975308642e-06, 'epoch': 20.67}                                           
{'embedding_loss': 0.0, 'learning_rate': 2.7777777777777783e-06, 'epoch': 21.0}                                          
{'embedding_loss': 0.0, 'learning_rate': 2.469135802469136e-06, 'epoch': 21.33}                                          
{'embedding_loss': 0.0, 'learning_rate': 2.1604938271604937e-06, 'epoch': 21.67}                                         
{'embedding_loss': 0.0, 'learning_rate': 1.8518518518518519e-06, 'epoch': 22.0}                                          
{'embedding_loss': 0.0, 'learning_rate': 1.54320987654321e-06, 'epoch': 22.33}                                           
{'embedding_loss': 0.0, 'learning_rate': 1.234567901234568e-06, 'epoch': 22.67}                                          
{'embedding_loss': 0.0, 'learning_rate': 9.259259259259259e-07, 'epoch': 23.0}                                           
{'embedding_loss': 0.0, 'learning_rate': 6.17283950617284e-07, 'epoch': 23.33}                                           
{'embedding_loss': 0.0001, 'learning_rate': 3.08641975308642e-07, 'epoch': 23.67}                                        
{'embedding_loss': 0.0, 'learning_rate': 0.0, 'epoch': 24.0}                                                             
{'train_runtime': 1231.5658, 'train_samples_per_second': 23.385, 'train_steps_per_second': 2.923, 'epoch': 24.0}         
100%|████████████████████████████████████████████████████████████████████████████████| 3600/3600 [20:31<00:00,  2.92it/s]
***** Running evaluation *****███████████████████████████████████████████████████████| 3600/3600 [20:31<00:00,  2.96it/s]

=== Evaluation ===
 {'accuracy': 0.9306930693069307}