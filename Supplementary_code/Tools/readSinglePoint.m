function M_all = readSinglePoint(lat,NS,ens_mems,var)
intro = 'b.e21.BWSSP245.f09_g17.release-cesm2.1.3.WACCM-MA-1deg.';
M_all = [];
for ens = ens_mems
    if ens == 3
        inj_name = ['INJANN' num2str(lat) NS '_12Tg'];
        intro = 'b.e21.BWSSP245.f09_g17.release-cesm2.1.5.WACCM-MA-1deg.';
        M = ncread(['Climate_Data/' inj_name '/'  intro inj_name '.003.cam.h0.' var '.203501-207012.nc'],var);
    elseif ens == 1 && lat~=60
        inj_name = ['INJANN' num2str(lat) NS '_12Tg'];
        if lat == 0
            M0 = ncread(['Climate_Data/' inj_name '/'  intro inj_name '.001.cam.h0.' var '.203501-204412.nc'],var);
            M1 = ncread(['Climate_Data/' inj_name '/'  intro inj_name '.001.cam.h0.' var '.204412-206412.nc'],var);
            M2 = ncread(['Climate_Data/' inj_name '/'  intro inj_name '.001.cam.h0.' var '.206412-206912.nc'],var);
            M1 = M1(:,:,2:end);
            M2 = M2(:,:,2:end);
            M = cat(3,M0,M1,M2);
            % if strcmp(var,'PRECT')
            % M = cat(3,ncread(['Climate_Data/' inj_name '/' 'PRECC' '_sysid_ens' num2str(lat) NS '.nc'],'PRECC') +ncread(['Climate_Data/' inj_name '/' 'PRECL' '_sysid_ens' num2str(lat) NS '.nc'],'PRECL'),cat(3,M1,M2));
            % else
            % M = cat(3,ncread(['Climate_Data/' inj_name '/' var '_sysid_ens' num2str(lat) NS '.nc'],var) ,cat(3,M1,M2));
            % end
        else
            M0 = ncread(['Climate_Data/' inj_name '/'  intro inj_name '.001.cam.h0.' var '.203501-204412.nc'],var);
            M1 = ncread(['Climate_Data/' inj_name '/'  intro inj_name '.001.cam.h0.' var '.204501-206412.nc'],var);
            M2 = ncread(['Climate_Data/' inj_name '/'  intro inj_name '.001.cam.h0.' var '.206412-206912.nc'],var);
            M2 = M2(:,:,2:end);
            M = cat(3,M0,M1,M2);
            % if strcmp(var,'PRECT')
            %     M = cat(3,ncread(['Climate_Data/' inj_name '/' 'PRECC' '_sysid_ens' num2str(lat) NS '.nc'],'PRECC') +ncread(['Climate_Data/' inj_name '/' 'PRECL' '_sysid_ens' num2str(lat) NS '.nc'],'PRECL'),cat(3,M1,M2));
            % else
            %     M = cat(3,ncread(['Climate_Data/' inj_name '/' var '_sysid_ens' num2str(lat) NS '.nc'],var) ,cat(3,M1,M2));
            % end
        end
    elseif lat == 60
        if strcmp(NS,'N')
            inj_name = 'INJMAM60N_12Tg';
        else
            inj_name = 'INJSON60S_12Tg';
        end
        M = ncread(['Climate_Data/' inj_name '/'  intro inj_name '.001.cam.h0.' var '.203501-206912.nc'],var);
    else
        inj_name = ['INJANN' num2str(lat) NS '_12Tg'];
        M = ncread(['Climate_Data/' inj_name '/'  intro inj_name '.002.cam.h0.' var '.203501-206912.nc'],var);
    end
    if ens == 3
        M = M(:,:,1:end-12);
    end
    M_all = cat(4,M_all,M);
end

end