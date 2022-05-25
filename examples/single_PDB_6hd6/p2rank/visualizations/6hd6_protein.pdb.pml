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

load "data/6hd6_protein.pdb", protein
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

load "data/6hd6_protein.pdb_points.pdb.gz", points
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
select surf_pocket1, protein and id [1260,1263,1267,1272,1275,1278,1280,1266,732,727,749,1187,1188,371,261,208,744,746,748,1274,209,210,236,238,239,241,211,598,548,592,596,550,599,1116,1104,1112,1270,1053,1054,1094,1097,496,522,523,524,712,497,498,499,728,709,385,387,383,262,492,514,515,495,776,777,780,787,808,811,752,766,767,760,850,230] 
set surface_color,  pcol1, surf_pocket1 
set_color pcol2 = [0.302,0.278,0.702]
select surf_pocket2, protein and id [904,939,1697,1699,1700,869,1930,1931,1678,1679,1681,1682,1920,914,915,918,937,917,2263,909,912,1942,942,2257,2287,941,1935,1966,2286,2321,1928,1929,2259,2262,1645,1676,973,935] 
set surface_color,  pcol2, surf_pocket2 
set_color pcol3 = [0.631,0.361,0.902]
select surf_pocket3, protein and id [1298,1299,1404,1112,1269,1270,1123,1125,1127,1284,1286,244,1274,1283,1285,1317,1318,1282,1266,458,435,436,457,463,464,386,387,384,391,469,467,249,250,247,251,497,498,499,402,1135,1426,1122,1427,1464,1134,1137,1448,1544,1545] 
set surface_color,  pcol3, surf_pocket3 
set_color pcol4 = [0.678,0.278,0.702]
select surf_pocket4, protein and id [840,1715,1718,1719,1344,1351,1352,1668,1709,1456,1726,1160,803,1457,1442,1723] 
set surface_color,  pcol4, surf_pocket4 
set_color pcol5 = [0.902,0.361,0.682]
select surf_pocket5, protein and id [583,584,986,987,988,567,569,1008,1009,1011,1032,1033,980,2157,2158,2159,2193,957,959,955,1034] 
set surface_color,  pcol5, surf_pocket5 
set_color pcol6 = [0.702,0.278,0.341]
select surf_pocket6, protein and id [820,822,858,899,846,853,772,818,771,773,774,1217,900,1224,1226,1199,1209,1210] 
set surface_color,  pcol6, surf_pocket6 
set_color pcol7 = [0.902,0.522,0.361]
select surf_pocket7, protein and id [1497,1381,1383,1385,1500,1375,1495,1496,1501,1502,1499,1796,1800,1769,1775,1776,1777,1802,1801,1803,1433,1435,1370] 
set surface_color,  pcol7, surf_pocket7 
set_color pcol8 = [0.702,0.596,0.278]
select surf_pocket8, protein and id [647,75,531,533,477,107,481,478,480,645,504,506,509] 
set surface_color,  pcol8, surf_pocket8 


deselect

orient
