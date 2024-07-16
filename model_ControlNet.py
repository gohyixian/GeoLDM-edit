Model running on device  : cuda
Model running on dtype   : torch.float32
Model Size               : 0.009599335491657257 GB  /  9.829719543457031 MB  /  10307208 Bytes
Training Dataset Name    : d_20240623_CrossDocked_LG_PKT__10A__CA_Only__no_H__LIGAND
Model Training Mode      : ControlNet
================================



ControlEnLatentDiffusion(
  (gamma): PredefinedNoiseSchedule()
  (dynamics): ControlNet_Module_Wrapper(
    (controlnet_arch_wrapper): ControlNet_Arch_Wrapper(
      (diffusion_net): EGNN(
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






      (control_net): EGNN(
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






      (fusion_net): EGNN_Fusion(
        (fusion_e_block_0): FusionBlock(
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
        (fusion_e_block_1): FusionBlock(
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
        (fusion_e_block_2): FusionBlock(
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
        (fusion_e_block_3): FusionBlock(
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






    (diffusion_network): EGNN_dynamics_QM9(
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






    (control_network): EGNN_dynamics_QM9(
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







    (fusion_network): EGNN_dynamics_fusion(
      (egnn_fusion): EGNN_Fusion(
        (fusion_e_block_0): FusionBlock(
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
        (fusion_e_block_1): FusionBlock(
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
        (fusion_e_block_2): FusionBlock(
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
        (fusion_e_block_3): FusionBlock(
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