function std_out = getPIstd(var_name,scaling)
string_p_1 = "Climate_Data/TSMLT_PI/b.e21.BWma1850.f09_g17.release-cesm2.1.3.c20200918.cam.h0.";
if strcmp(var_name,"PRECT")
    var_name = "PRECL";
    var2 = "PRECC";
part_1 = ncread([string_p_1 + var_name + ".011001-015912.nc"],var_name)+ncread([string_p_1 + var2 + ".011001-015912.nc"],var2);
part_2 = ncread([string_p_1 + var_name + ".016001-020912.nc"],var_name)+ncread([string_p_1 + var2 + ".016001-020912.nc"],var2);
part_3 = ncread([string_p_1 + var_name + ".021001-021612.nc"],var_name)+ ncread([string_p_1 + var2 + ".021001-021612.nc"],var2);
part_4 = ncread([string_p_1 + var_name + ".021701-026212.nc"],var_name)+ncread([string_p_1 + var2 + ".021701-026212.nc"],var2);
% part_5 = ncread([string_p_1 + var_name + ".035001-039912.nc"],var_name)+ncread([string_p_1 + var2 + ".035001-039912.nc"],var2);
% part_6 = ncread([string_p_1 + var_name + ".040001-044912.nc"],var_name)+ncread([string_p_1 + var2 + ".040001-044912.nc"],var2);
% part_7 = ncread([string_p_1 + var_name + ".045001-049912.nc"],var_name)+ncread([string_p_1 + var2 + ".045001-049912.nc"],var2);
else
part_1 = ncread([string_p_1 + var_name + ".011001-015912.nc"],var_name);
part_2 = ncread([string_p_1 + var_name + ".016001-020912.nc"],var_name);
part_3 = ncread([string_p_1 + var_name + ".021001-021612.nc"],var_name);
part_4 = ncread([string_p_1 + var_name + ".021701-026212.nc"],var_name);
% part_5 = ncread([string_p_1 + var_name + ".035001-039912.nc"],var_name);
% part_6 = ncread([string_p_1 + var_name + ".040001-044912.nc"],var_name);
% part_7 = ncread([string_p_1 + var_name + ".045001-049912.nc"],var_name);
   
end

full_PI_run = scaling*cat(3,part_1,part_2,part_3,part_4);

global_mean_all_parts = globalMean(full_PI_run);
ann_mean = averageEvery2d(12,1,global_mean_all_parts);
all_variability = ann_mean-mean(ann_mean);

full_PI_run_ann = averageEvery(12,1,full_PI_run);
std_out = std(full_PI_run_ann,0,3);
end