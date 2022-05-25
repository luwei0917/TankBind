from pymol import cmd,stored

set depth_cue, 1
set fog_start, 0.4

set_color b_col, [36,36,85]
set_color t_col, [10,10,10]
set bg_rgb_bottom, b_col
set bg_rgb_top, t_col      
set bg_gradient

set  spec_power  =  200
set  spec_refl   =  0

load "data/6dlo.pdb", protein
create ligands, protein and organic
select xlig, protein and organic
delete xlig

hide everything, all

color white, elem c
color bluewhite, protein
#show_as cartoon, protein
show surface, protein
#set transparency, 0.15

show sticks, ligands
set stick_color, magenta

load "data/6dlo.pdb_points.pdb.gz", points
hide nonbonded, points
show nb_spheres, points
set sphere_scale, 0.2, points
cmd.spectrum("b", "green_red", selection="points", minimum=0, maximum=0.7)


stored.list=[]
cmd.iterate("(resn STP)","stored.list.append(resi)")    # read info about residues STP
lastSTP=stored.list[-1] # get the index of the last residue
hide lines, resn STP

cmd.select("rest", "resn STP and resi 0")

for my_index in range(1,int(lastSTP)+1): cmd.select("pocket"+str(my_index), "resn STP and resi "+str(my_index))
for my_index in range(1,int(lastSTP)+1): cmd.show("spheres","pocket"+str(my_index))
for my_index in range(1,int(lastSTP)+1): cmd.set("sphere_scale","0.4","pocket"+str(my_index))
for my_index in range(1,int(lastSTP)+1): cmd.set("sphere_transparency","0.1","pocket"+str(my_index))



set_color pcol1 = [0.361,0.576,0.902]
select surf_pocket1, protein and id [3926,4314,4638,3927,3941,3917,3918,3565,4324,4325,4328,3180,3181,3560,3920,3556,3185,4308,4309,4620,4634,3921,4298,2827,4019,2507,2511,4018,4020,2830,4017,3932,3580,3562,3567,3570,4329,3189,4647,4649,4650,4654,4651,4655,4656,2835,3193,2522,2840,2837,2533,3206,3204,4633,2517,2514,2519,2826] 
set surface_color,  pcol1, surf_pocket1 
set_color pcol2 = [0.278,0.278,0.702]
select surf_pocket2, protein and id [4650,2534,2537,2538,3193,2522,2840,2849,2533,2846,2850,2853,2854,3206,3204,3210,3213,3214,3215] 
set surface_color,  pcol2, surf_pocket2 
set_color pcol3 = [0.576,0.361,0.902]
select surf_pocket3, protein and id [2585,3066,2767,2764,3032,2661,2665,2676,2677,2678,2584,2672,2679,2680,2681,2682,2750,2752,3044,3051,3065,3031] 
set surface_color,  pcol3, surf_pocket3 
set_color pcol4 = [0.616,0.278,0.702]
select surf_pocket4, protein and id [4417,4420,4431,4433,4577,4263,4432,4415,4599,4414,4416,4413,4271,4272,4252,4267,4261,4282,4273] 
set surface_color,  pcol4, surf_pocket4 
set_color pcol5 = [0.902,0.361,0.792]
select surf_pocket5, protein and id [2656,2657,2759,2616,2617,2618,2462,2741,2743,2476,2488,2468] 
set surface_color,  pcol5, surf_pocket5 
set_color pcol6 = [0.702,0.278,0.447]
select surf_pocket6, protein and id [2827,2829,2830,2983,3180,3181,3543,3558,3559,3560,3542,3316,3169,3183] 
set surface_color,  pcol6, surf_pocket6 
set_color pcol7 = [0.902,0.361,0.361]
select surf_pocket7, protein and id [4479,3963,3987,3988,4070,3969,4475,4476,4514,4477,4485,4149,4152,4147,4148,4150,4469] 
set surface_color,  pcol7, surf_pocket7 
set_color pcol8 = [0.702,0.447,0.278]
select surf_pocket8, protein and id [3652,3537,3642,3779,3637,3778,3634,3530,3528,3904,3643,3890] 
set surface_color,  pcol8, surf_pocket8 
set_color pcol9 = [0.902,0.792,0.361]
select surf_pocket9, protein and id [3046,3047,2928,2929,3056,2856,2857,2544,2545,2871] 
set surface_color,  pcol9, surf_pocket9 


deselect

orient
