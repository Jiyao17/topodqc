Minimize
 obj: - 2 lambda_bin_0 - 4 lambda_bin_1 - 8 lambda_bin_2 - 16 lambda_bin_3 
 - 32 lambda_bin_4 - 64 lambda_bin_5 - 128 lambda_bin_6 - 256 lambda_bin_7 
 + 0.0625 lambda_bin_8 + 0.125 lambda_bin_9 + 0.25 lambda_bin_10 
 + 0.5 lambda_bin_11 + 1 lambda_bin_12 + 2 lambda_bin_13 + 4 lambda_bin_14 
 + 8 lambda_bin_15 + 16 lambda_bin_16 + 32 lambda_bin_17 + 64 lambda_bin_18 
 + 128 lambda_bin_19 + 256 lambda_bin_20 + 512 lambda_bin_21 
 + 1024 lambda_bin_22 + 2048 lambda_bin_23 + 4096 lambda_bin_24 
 + 8192 lambda_bin_25 + 16384 lambda_bin_26 

Subject To 
 CQM_MAP_constraint_0: + 1 bin_0 + 1 bin_1 + 1 bin_2 + 1 bin_3  = 1.0
 CQM_MAP_constraint_1: + 1 bin_4 + 1 bin_5 + 1 bin_6 + 1 bin_7  = 1.0
 CQM_MAP_constraint_2: + 1 bin_8 + 1 bin_9 + 1 bin_10 + 1 bin_11  = 1.0
 CQM_MAP_constraint_3: + 1 bin_12 + 1 bin_13 + 1 bin_14 + 1 bin_15  = 1.0
 CQM_MAP_constraint_4: + 1 bin_16 + 1 bin_17 + 1 bin_18 + 1 bin_19  = 1.0
 CQM_MAP_constraint_5: + 1 bin_20 + 1 bin_21 + 1 bin_22 + 1 bin_23  = 1.0
 CQM_MAP_constraint_6: + 1 bin_24 + 1 bin_25 + 1 bin_26 + 1 bin_27  = 1.0
 CQM_MAP_constraint_7: + 1 bin_28 + 1 bin_29 + 1 bin_30 + 1 bin_31  = 1.0
 CQM_MAP_constraint_8: + 1 bin_32 + 1 bin_33 + 1 bin_34 + 1 bin_35  = 1.0
 CQM_MAP_constraint_9: + 1 bin_36 + 1 bin_37 + 1 bin_38 + 1 bin_39  = 1.0
 CQM_MAP_constraint_10: + 1 bin_40 + 1 bin_41 + 1 bin_42 + 1 bin_43  = 1.0
 CQM_MAP_constraint_11: + 1 bin_44 + 1 bin_45 + 1 bin_46 + 1 bin_47  = 1.0
 CQM_MAP_constraint_12: + 1 bin_48 + 1 bin_49 + 1 bin_50 + 1 bin_51  = 1.0
 CQM_MAP_constraint_13: + 1 bin_52 + 1 bin_53 + 1 bin_54 + 1 bin_55  = 1.0
 CQM_MAP_constraint_14: - 1 bin_0 - 1 bin_4 - 1 bin_8 - 1 bin_12 - 1 bin_16 
 - 1 bin_20 - 1 bin_24 - 1 bin_28 - 1 bin_32 - 1 bin_36 - 1 bin_40 - 1 bin_44 
 - 1 bin_48 - 1 bin_52  >= -4.0
 CQM_MAP_constraint_15: - 1 bin_1 - 1 bin_5 - 1 bin_9 - 1 bin_13 - 1 bin_17 
 - 1 bin_21 - 1 bin_25 - 1 bin_29 - 1 bin_33 - 1 bin_37 - 1 bin_41 - 1 bin_45 
 - 1 bin_49 - 1 bin_53  >= -4.0
 CQM_MAP_constraint_16: - 1 bin_2 - 1 bin_6 - 1 bin_10 - 1 bin_14 - 1 bin_18 
 - 1 bin_22 - 1 bin_26 - 1 bin_30 - 1 bin_34 - 1 bin_38 - 1 bin_42 - 1 bin_46 
 - 1 bin_50 - 1 bin_54  >= -4.0
 CQM_MAP_constraint_17: - 1 bin_3 - 1 bin_7 - 1 bin_11 - 1 bin_15 - 1 bin_19 
 - 1 bin_23 - 1 bin_27 - 1 bin_31 - 1 bin_35 - 1 bin_39 - 1 bin_43 - 1 bin_47 
 - 1 bin_51 - 1 bin_55  >= -4.0
 CQM_MAP_constraint_18: + 1 bin_56 + 1 bin_57 + 1 bin_58 - 1 bin_59 - 1 bin_62 
 - 1 bin_65 - 1 bin_128  = 0.0
 CQM_MAP_constraint_19: - 1 bin_56 + 1 bin_59 + 1 bin_60 + 1 bin_61 - 1 bin_63 
 - 1 bin_66 + 1 bin_128  = 0.0
 CQM_MAP_constraint_20: - 1 bin_57 - 1 bin_60 + 1 bin_62 + 1 bin_63 + 1 bin_64 
 - 1 bin_67  = 0.0
 CQM_MAP_constraint_21: - 1 bin_58 - 1 bin_61 - 1 bin_64 + 1 bin_65 + 1 bin_66 
 + 1 bin_67  = 0.0
 CQM_MAP_constraint_22: + 1 bin_68 + 1 bin_69 + 1 bin_70 - 1 bin_71 - 1 bin_74 
 - 1 bin_77 - 1 bin_129  = 0.0
 CQM_MAP_constraint_23: - 1 bin_69 - 1 bin_72 + 1 bin_74 + 1 bin_75 + 1 bin_76 
 - 1 bin_79 + 1 bin_129  = 0.0
 CQM_MAP_constraint_24: - 1 bin_68 + 1 bin_71 + 1 bin_72 + 1 bin_73 - 1 bin_75 
 - 1 bin_78  = 0.0
 CQM_MAP_constraint_25: - 1 bin_70 - 1 bin_73 - 1 bin_76 + 1 bin_77 + 1 bin_78 
 + 1 bin_79  = 0.0
 CQM_MAP_constraint_26: + 1 bin_80 + 1 bin_81 + 1 bin_82 - 1 bin_83 - 1 bin_86 
 - 1 bin_89 - 1 bin_130  = 0.0
 CQM_MAP_constraint_27: - 1 bin_82 - 1 bin_85 - 1 bin_88 + 1 bin_89 + 1 bin_90 
 + 1 bin_91 + 1 bin_130  = 0.0
 CQM_MAP_constraint_28: - 1 bin_80 + 1 bin_83 + 1 bin_84 + 1 bin_85 - 1 bin_87 
 - 1 bin_90  = 0.0
 CQM_MAP_constraint_29: - 1 bin_81 - 1 bin_84 + 1 bin_86 + 1 bin_87 + 1 bin_88 
 - 1 bin_91  = 0.0
 CQM_MAP_constraint_30: - 1 bin_92 + 1 bin_95 + 1 bin_96 + 1 bin_97 - 1 bin_99 
 - 1 bin_102 - 1 bin_131  = 0.0
 CQM_MAP_constraint_31: - 1 bin_93 - 1 bin_96 + 1 bin_98 + 1 bin_99 
 + 1 bin_100 - 1 bin_103 + 1 bin_131  = 0.0
 CQM_MAP_constraint_32: + 1 bin_92 + 1 bin_93 + 1 bin_94 - 1 bin_95 - 1 bin_98 
 - 1 bin_101  = 0.0
 CQM_MAP_constraint_33: - 1 bin_94 - 1 bin_97 - 1 bin_100 + 1 bin_101 
 + 1 bin_102 + 1 bin_103  = 0.0
 CQM_MAP_constraint_34: - 1 bin_104 + 1 bin_107 + 1 bin_108 + 1 bin_109 
 - 1 bin_111 - 1 bin_114 - 1 bin_132  = 0.0
 CQM_MAP_constraint_35: - 1 bin_106 - 1 bin_109 - 1 bin_112 + 1 bin_113 
 + 1 bin_114 + 1 bin_115 + 1 bin_132  = 0.0
 CQM_MAP_constraint_36: + 1 bin_104 + 1 bin_105 + 1 bin_106 - 1 bin_107 
 - 1 bin_110 - 1 bin_113  = 0.0
 CQM_MAP_constraint_37: - 1 bin_105 - 1 bin_108 + 1 bin_110 + 1 bin_111 
 + 1 bin_112 - 1 bin_115  = 0.0
 CQM_MAP_constraint_38: - 1 bin_117 - 1 bin_120 + 1 bin_122 + 1 bin_123 
 + 1 bin_124 - 1 bin_127 - 1 bin_133  = 0.0
 CQM_MAP_constraint_39: - 1 bin_118 - 1 bin_121 - 1 bin_124 + 1 bin_125 
 + 1 bin_126 + 1 bin_127 + 1 bin_133  = 0.0
 CQM_MAP_constraint_40: + 1 bin_116 + 1 bin_117 + 1 bin_118 - 1 bin_119 
 - 1 bin_122 - 1 bin_125  = 0.0
 CQM_MAP_constraint_41: - 1 bin_116 + 1 bin_119 + 1 bin_120 + 1 bin_121 
 - 1 bin_123 - 1 bin_126  = 0.0
 CQM_MAP_constraint_42: - 1 bin_56 - 1 bin_57 - 1 bin_58 - 1 bin_59 - 1 bin_60 
 - 1 bin_61 - 1 bin_62 - 1 bin_63 - 1 bin_64 - 1 bin_65 - 1 bin_66 - 1 bin_67 
 + 3 bin_128  >= -0.0
 CQM_MAP_constraint_43: - 1 bin_68 - 1 bin_69 - 1 bin_70 - 1 bin_71 - 1 bin_72 
 - 1 bin_73 - 1 bin_74 - 1 bin_75 - 1 bin_76 - 1 bin_77 - 1 bin_78 - 1 bin_79 
 + 3 bin_129  >= -0.0
 CQM_MAP_constraint_44: - 1 bin_80 - 1 bin_81 - 1 bin_82 - 1 bin_83 - 1 bin_84 
 - 1 bin_85 - 1 bin_86 - 1 bin_87 - 1 bin_88 - 1 bin_89 - 1 bin_90 - 1 bin_91 
 + 3 bin_130  >= -0.0
 CQM_MAP_constraint_45: - 1 bin_92 - 1 bin_93 - 1 bin_94 - 1 bin_95 - 1 bin_96 
 - 1 bin_97 - 1 bin_98 - 1 bin_99 - 1 bin_100 - 1 bin_101 - 1 bin_102 
 - 1 bin_103 + 3 bin_131  >= -0.0
 CQM_MAP_constraint_46: - 1 bin_104 - 1 bin_105 - 1 bin_106 - 1 bin_107 
 - 1 bin_108 - 1 bin_109 - 1 bin_110 - 1 bin_111 - 1 bin_112 - 1 bin_113 
 - 1 bin_114 - 1 bin_115 + 3 bin_132  >= -0.0
 CQM_MAP_constraint_47: - 1 bin_116 - 1 bin_117 - 1 bin_118 - 1 bin_119 
 - 1 bin_120 - 1 bin_121 - 1 bin_122 - 1 bin_123 - 1 bin_124 - 1 bin_125 
 - 1 bin_126 - 1 bin_127 + 3 bin_133  >= -0.0
 CQM_MAP_constraint_48: + 1 bin_56 + 1 bin_59 + 1 bin_68 + 1 bin_71 + 1 bin_80 
 + 1 bin_83 + 1 bin_92 + 1 bin_95 + 1 bin_104 + 1 bin_107 + 1 bin_116 
 + 1 bin_119 - 1 bin_134  >= -0.0
 CQM_MAP_constraint_49: - 1 bin_56 - 1 bin_59 - 1 bin_68 - 1 bin_71 - 1 bin_80 
 - 1 bin_83 - 1 bin_92 - 1 bin_95 - 1 bin_104 - 1 bin_107 - 1 bin_116 
 - 1 bin_119 + 12 bin_134  >= -0.0
 CQM_MAP_constraint_50: + 1 bin_57 + 1 bin_62 + 1 bin_69 + 1 bin_74 + 1 bin_81 
 + 1 bin_86 + 1 bin_93 + 1 bin_98 + 1 bin_105 + 1 bin_110 + 1 bin_117 
 + 1 bin_122 - 1 bin_135  >= -0.0
 CQM_MAP_constraint_51: - 1 bin_57 - 1 bin_62 - 1 bin_69 - 1 bin_74 - 1 bin_81 
 - 1 bin_86 - 1 bin_93 - 1 bin_98 - 1 bin_105 - 1 bin_110 - 1 bin_117 
 - 1 bin_122 + 12 bin_135  >= -0.0
 CQM_MAP_constraint_52: + 1 bin_58 + 1 bin_65 + 1 bin_70 + 1 bin_77 + 1 bin_82 
 + 1 bin_89 + 1 bin_94 + 1 bin_101 + 1 bin_106 + 1 bin_113 + 1 bin_118 
 + 1 bin_125 - 1 bin_136  >= -0.0
 CQM_MAP_constraint_53: - 1 bin_58 - 1 bin_65 - 1 bin_70 - 1 bin_77 - 1 bin_82 
 - 1 bin_89 - 1 bin_94 - 1 bin_101 - 1 bin_106 - 1 bin_113 - 1 bin_118 
 - 1 bin_125 + 12 bin_136  >= -0.0
 CQM_MAP_constraint_54: + 1 bin_60 + 1 bin_63 + 1 bin_72 + 1 bin_75 + 1 bin_84 
 + 1 bin_87 + 1 bin_96 + 1 bin_99 + 1 bin_108 + 1 bin_111 + 1 bin_120 
 + 1 bin_123 - 1 bin_137  >= -0.0
 CQM_MAP_constraint_55: - 1 bin_60 - 1 bin_63 - 1 bin_72 - 1 bin_75 - 1 bin_84 
 - 1 bin_87 - 1 bin_96 - 1 bin_99 - 1 bin_108 - 1 bin_111 - 1 bin_120 
 - 1 bin_123 + 12 bin_137  >= -0.0
 CQM_MAP_constraint_56: + 1 bin_61 + 1 bin_66 + 1 bin_73 + 1 bin_78 + 1 bin_85 
 + 1 bin_90 + 1 bin_97 + 1 bin_102 + 1 bin_109 + 1 bin_114 + 1 bin_121 
 + 1 bin_126 - 1 bin_138  >= -0.0
 CQM_MAP_constraint_57: - 1 bin_61 - 1 bin_66 - 1 bin_73 - 1 bin_78 - 1 bin_85 
 - 1 bin_90 - 1 bin_97 - 1 bin_102 - 1 bin_109 - 1 bin_114 - 1 bin_121 
 - 1 bin_126 + 12 bin_138  >= -0.0
 CQM_MAP_constraint_58: + 1 bin_64 + 1 bin_67 + 1 bin_76 + 1 bin_79 + 1 bin_88 
 + 1 bin_91 + 1 bin_100 + 1 bin_103 + 1 bin_112 + 1 bin_115 + 1 bin_124 
 + 1 bin_127 - 1 bin_139  >= -0.0
 CQM_MAP_constraint_59: - 1 bin_64 - 1 bin_67 - 1 bin_76 - 1 bin_79 - 1 bin_88 
 - 1 bin_91 - 1 bin_100 - 1 bin_103 - 1 bin_112 - 1 bin_115 - 1 bin_124 
 - 1 bin_127 + 12 bin_139  >= -0.0
 CQM_MAP_constraint_60: - 1 bin_134 - 1 bin_135 - 1 bin_136 - 1 bin_137 
 - 1 bin_138 - 1 bin_139  >= -6.0
 CQM_MAP_constraint_61: - 1 bin_134 - 1 bin_135 - 1 bin_136  >= -4.0
 CQM_MAP_constraint_62: - 1 bin_134 - 1 bin_137 - 1 bin_138  >= -4.0
 CQM_MAP_constraint_63: - 1 bin_135 - 1 bin_137 - 1 bin_139  >= -4.0
 CQM_MAP_constraint_64: - 1 bin_136 - 1 bin_138 - 1 bin_139  >= -4.0

Bounds

Binary
 bin_0 bin_1 bin_2 bin_3 bin_4 bin_5 bin_6 bin_7 bin_8 bin_9 bin_10 bin_11
  bin_12 bin_13 bin_14 bin_15 bin_16 bin_17 bin_18 bin_19 bin_20 bin_21 bin_22
  bin_23 bin_24 bin_25 bin_26 bin_27 bin_28 bin_29 bin_30 bin_31 bin_32 bin_33
  bin_34 bin_35 bin_36 bin_37 bin_38 bin_39 bin_40 bin_41 bin_42 bin_43 bin_44
  bin_45 bin_46 bin_47 bin_48 bin_49 bin_50 bin_51 bin_52 bin_53 bin_54 bin_55
  bin_56 bin_57 bin_58 bin_59 bin_60 bin_61 bin_62 bin_63 bin_64 bin_65 bin_66
  bin_67 bin_68 bin_69 bin_70 bin_71 bin_72 bin_73 bin_74 bin_75 bin_76 bin_77
  bin_78 bin_79 bin_80 bin_81 bin_82 bin_83 bin_84 bin_85 bin_86 bin_87 bin_88
  bin_89 bin_90 bin_91 bin_92 bin_93 bin_94 bin_95 bin_96 bin_97 bin_98 bin_99
  bin_100 bin_101 bin_102 bin_103 bin_104 bin_105 bin_106 bin_107 bin_108
  bin_109 bin_110 bin_111 bin_112 bin_113 bin_114 bin_115 bin_116 bin_117
  bin_118 bin_119 bin_120 bin_121 bin_122 bin_123 bin_124 bin_125 bin_126
  bin_127 bin_128 bin_129 bin_130 bin_131 bin_132 bin_133 bin_134 bin_135
  bin_136 bin_137 bin_138 bin_139 lambda_bin_0 lambda_bin_1 lambda_bin_2
  lambda_bin_3 lambda_bin_4 lambda_bin_5 lambda_bin_6 lambda_bin_7 lambda_bin_8
  lambda_bin_9 lambda_bin_10 lambda_bin_11 lambda_bin_12 lambda_bin_13
  lambda_bin_14 lambda_bin_15 lambda_bin_16 lambda_bin_17 lambda_bin_18
  lambda_bin_19 lambda_bin_20 lambda_bin_21 lambda_bin_22 lambda_bin_23
  lambda_bin_24 lambda_bin_25 lambda_bin_26
General

End