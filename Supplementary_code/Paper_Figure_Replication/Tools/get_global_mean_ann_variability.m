function variability = get_global_mean_ann_variability(var,duration,n)


part_1 = ncread(["Climate_Data/TSMLT_PI/b.e21.BW1850.f09_g17.CMIP6-piControl.001.cam.h0." + var + ".000101-009912.nc"],var);
part_2 = ncread(["Climate_Data/TSMLT_PI/b.e21.BW1850.f09_g17.CMIP6-piControl.001.cam.h0." + var + ".010001-019912.nc"],var);
part_3 = ncread(["Climate_Data/TSMLT_PI/b.e21.BW1850.f09_g17.CMIP6-piControl.001.cam.h0." + var + ".020001-029912.nc"],var);
part_4 = ncread(["Climate_Data/TSMLT_PI/b.e21.BW1850.f09_g17.CMIP6-piControl.001.cam.h0." + var + ".030001-034912.nc"],var);
part_5 = ncread(["Climate_Data/TSMLT_PI/b.e21.BW1850.f09_g17.CMIP6-piControl.001.cam.h0." + var + ".035001-039912.nc"],var);
part_6 = ncread(["Climate_Data/TSMLT_PI/b.e21.BW1850.f09_g17.CMIP6-piControl.001.cam.h0." + var + ".040001-044912.nc"],var);
part_7 = ncread(["Climate_Data/TSMLT_PI/b.e21.BW1850.f09_g17.CMIP6-piControl.001.cam.h0." + var + ".045001-049912.nc"],var);

full_PI_run = cat(3,part_1,part_2,part_3,part_4,part_5,part_6,part_7);

global_mean_all_parts = globalMean(full_PI_run);
ann_mean = averageEvery2d(12,1,global_mean_all_parts);
all_variability = ann_mean-mean(ann_mean);
random_index = randi(length(all_variability)-duration+1,n,1);

variability = zeros(duration,n);
for i = 1:n
    variability(:,i) = all_variability(random_index(i):random_index(i)+duration-1);
end

end