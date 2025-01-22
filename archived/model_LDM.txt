Model running on device  : cuda
Model running on dtype   : torch.float32
Model Size               : 0.005155649036169052 GB  /  5.279384613037109 MB  /  5535836 Bytes
Training Dataset Name    : d_20240623_CrossDocked_LG_PKT__10A__CA_Only__no_H__LIGAND+POCKET
Model Training Mode      : LDM
================================



EnLatentDiffusion(
  (gamma): PredefinedNoiseSchedule()
  (dynamics): EGNN_dynamics_QM9(
    (egnn): EGNN(
      (embedding): Linear(in_features=3, out_features=128, bias=True)
      (embedding_out): Linear(in_features=128, out_features=3, bias=True)
      (e_block_0): EquivariantBlock(
        (gcl_0): GCL(
          (edge_mlp): Sequential(
            (0): Linear(in_features=258, out_features=128, bias=True)
            (1): SiLU()
            (2): Linear(in_features=128, out_features=128, bias=True)
            (3): SiLU()
          )
          (node_mlp): Sequential(
            (0): Linear(in_features=256, out_features=128, bias=True)
            (1): SiLU()
            (2): Linear(in_features=128, out_features=128, bias=True)
          )
          (att_mlp): Sequential(
            (0): Linear(in_features=128, out_features=1, bias=True)
            (1): Sigmoid()
          )
        )
        (gcl_equiv): EquivariantUpdate(
          (coord_mlp): Sequential(
            (0): Linear(in_features=258, out_features=128, bias=True)
            (1): SiLU()
            (2): Linear(in_features=128, out_features=128, bias=True)
            (3): SiLU()
            (4): Linear(in_features=128, out_features=1, bias=False)
          )
        )
      )
      (e_block_1): EquivariantBlock(
        (gcl_0): GCL(
          (edge_mlp): Sequential(
            (0): Linear(in_features=258, out_features=128, bias=True)
            (1): SiLU()
            (2): Linear(in_features=128, out_features=128, bias=True)
            (3): SiLU()
          )
          (node_mlp): Sequential(
            (0): Linear(in_features=256, out_features=128, bias=True)
            (1): SiLU()
            (2): Linear(in_features=128, out_features=128, bias=True)
          )
          (att_mlp): Sequential(
            (0): Linear(in_features=128, out_features=1, bias=True)
            (1): Sigmoid()
          )
        )
        (gcl_equiv): EquivariantUpdate(
          (coord_mlp): Sequential(
            (0): Linear(in_features=258, out_features=128, bias=True)
            (1): SiLU()
            (2): Linear(in_features=128, out_features=128, bias=True)
            (3): SiLU()
            (4): Linear(in_features=128, out_features=1, bias=False)
          )
        )
      )
      (e_block_2): EquivariantBlock(
        (gcl_0): GCL(
          (edge_mlp): Sequential(
            (0): Linear(in_features=258, out_features=128, bias=True)
            (1): SiLU()
            (2): Linear(in_features=128, out_features=128, bias=True)
            (3): SiLU()
          )
          (node_mlp): Sequential(
            (0): Linear(in_features=256, out_features=128, bias=True)
            (1): SiLU()
            (2): Linear(in_features=128, out_features=128, bias=True)
          )
          (att_mlp): Sequential(
            (0): Linear(in_features=128, out_features=1, bias=True)
            (1): Sigmoid()
          )
        )
        (gcl_equiv): EquivariantUpdate(
          (coord_mlp): Sequential(
            (0): Linear(in_features=258, out_features=128, bias=True)
            (1): SiLU()
            (2): Linear(in_features=128, out_features=128, bias=True)
            (3): SiLU()
            (4): Linear(in_features=128, out_features=1, bias=False)
          )
        )
      )
      (e_block_3): EquivariantBlock(
        (gcl_0): GCL(
          (edge_mlp): Sequential(
            (0): Linear(in_features=258, out_features=128, bias=True)
            (1): SiLU()
            (2): Linear(in_features=128, out_features=128, bias=True)
            (3): SiLU()
          )
          (node_mlp): Sequential(
            (0): Linear(in_features=256, out_features=128, bias=True)
            (1): SiLU()
            (2): Linear(in_features=128, out_features=128, bias=True)
          )
          (att_mlp): Sequential(
            (0): Linear(in_features=128, out_features=1, bias=True)
            (1): Sigmoid()
          )
        )
        (gcl_equiv): EquivariantUpdate(
          (coord_mlp): Sequential(
            (0): Linear(in_features=258, out_features=128, bias=True)
            (1): SiLU()
            (2): Linear(in_features=128, out_features=128, bias=True)
            (3): SiLU()
            (4): Linear(in_features=128, out_features=1, bias=False)
          )
        )
      )
    )
  )
  
  
  
  (vae): EnHierarchicalVAE(
    (encoder): EGNN_encoder_QM9(
      (egnn): EGNN(
        (embedding): Linear(in_features=27, out_features=128, bias=True)
        (embedding_out): Linear(in_features=128, out_features=128, bias=True)
        (e_block_0): EquivariantBlock(
          (gcl_0): GCL(
            (edge_mlp): Sequential(
              (0): Linear(in_features=258, out_features=128, bias=True)
              (1): SiLU()
              (2): Linear(in_features=128, out_features=128, bias=True)
              (3): SiLU()
            )
            (node_mlp): Sequential(
              (0): Linear(in_features=256, out_features=128, bias=True)
              (1): SiLU()
              (2): Linear(in_features=128, out_features=128, bias=True)
            )
            (att_mlp): Sequential(
              (0): Linear(in_features=128, out_features=1, bias=True)
              (1): Sigmoid()
            )
          )
          (gcl_equiv): EquivariantUpdate(
            (coord_mlp): Sequential(
              (0): Linear(in_features=258, out_features=128, bias=True)
              (1): SiLU()
              (2): Linear(in_features=128, out_features=128, bias=True)
              (3): SiLU()
              (4): Linear(in_features=128, out_features=1, bias=False)
            )
          )
        )
      )
      (final_mlp): Sequential(
        (0): Linear(in_features=128, out_features=128, bias=True)
        (1): SiLU()
        (2): Linear(in_features=128, out_features=5, bias=True)
      )
    )
    
    
    
    
    (decoder): EGNN_decoder_QM9(
      (egnn): EGNN(
        (embedding): Linear(in_features=2, out_features=128, bias=True)
        (embedding_out): Linear(in_features=128, out_features=27, bias=True)
        (e_block_0): EquivariantBlock(
          (gcl_0): GCL(
            (edge_mlp): Sequential(
              (0): Linear(in_features=258, out_features=128, bias=True)
              (1): SiLU()
              (2): Linear(in_features=128, out_features=128, bias=True)
              (3): SiLU()
            )
            (node_mlp): Sequential(
              (0): Linear(in_features=256, out_features=128, bias=True)
              (1): SiLU()
              (2): Linear(in_features=128, out_features=128, bias=True)
            )
            (att_mlp): Sequential(
              (0): Linear(in_features=128, out_features=1, bias=True)
              (1): Sigmoid()
            )
          )
          (gcl_equiv): EquivariantUpdate(
            (coord_mlp): Sequential(
              (0): Linear(in_features=258, out_features=128, bias=True)
              (1): SiLU()
              (2): Linear(in_features=128, out_features=128, bias=True)
              (3): SiLU()
              (4): Linear(in_features=128, out_features=1, bias=False)
            )
          )
        )
        (e_block_1): EquivariantBlock(
          (gcl_0): GCL(
            (edge_mlp): Sequential(
              (0): Linear(in_features=258, out_features=128, bias=True)
              (1): SiLU()
              (2): Linear(in_features=128, out_features=128, bias=True)
              (3): SiLU()
            )
            (node_mlp): Sequential(
              (0): Linear(in_features=256, out_features=128, bias=True)
              (1): SiLU()
              (2): Linear(in_features=128, out_features=128, bias=True)
            )
            (att_mlp): Sequential(
              (0): Linear(in_features=128, out_features=1, bias=True)
              (1): Sigmoid()
            )
          )
          (gcl_equiv): EquivariantUpdate(
            (coord_mlp): Sequential(
              (0): Linear(in_features=258, out_features=128, bias=True)
              (1): SiLU()
              (2): Linear(in_features=128, out_features=128, bias=True)
              (3): SiLU()
              (4): Linear(in_features=128, out_features=1, bias=False)
            )
          )
        )
        (e_block_2): EquivariantBlock(
          (gcl_0): GCL(
            (edge_mlp): Sequential(
              (0): Linear(in_features=258, out_features=128, bias=True)
              (1): SiLU()
              (2): Linear(in_features=128, out_features=128, bias=True)
              (3): SiLU()
            )
            (node_mlp): Sequential(
              (0): Linear(in_features=256, out_features=128, bias=True)
              (1): SiLU()
              (2): Linear(in_features=128, out_features=128, bias=True)
            )
            (att_mlp): Sequential(
              (0): Linear(in_features=128, out_features=1, bias=True)
              (1): Sigmoid()
            )
          )
          (gcl_equiv): EquivariantUpdate(
            (coord_mlp): Sequential(
              (0): Linear(in_features=258, out_features=128, bias=True)
              (1): SiLU()
              (2): Linear(in_features=128, out_features=128, bias=True)
              (3): SiLU()
              (4): Linear(in_features=128, out_features=1, bias=False)
            )
          )
        )
        (e_block_3): EquivariantBlock(
          (gcl_0): GCL(
            (edge_mlp): Sequential(
              (0): Linear(in_features=258, out_features=128, bias=True)
              (1): SiLU()
              (2): Linear(in_features=128, out_features=128, bias=True)
              (3): SiLU()
            )
            (node_mlp): Sequential(
              (0): Linear(in_features=256, out_features=128, bias=True)
              (1): SiLU()
              (2): Linear(in_features=128, out_features=128, bias=True)
            )
            (att_mlp): Sequential(
              (0): Linear(in_features=128, out_features=1, bias=True)
              (1): Sigmoid()
            )
          )
          (gcl_equiv): EquivariantUpdate(
            (coord_mlp): Sequential(
              (0): Linear(in_features=258, out_features=128, bias=True)
              (1): SiLU()
              (2): Linear(in_features=128, out_features=128, bias=True)
              (3): SiLU()
              (4): Linear(in_features=128, out_features=1, bias=False)
            )
          )
        )
      )
    )
  )
)