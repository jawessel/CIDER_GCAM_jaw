%% Intro
fullclear
addAllPaths
p = [];
p.var = 'TREFHT';
p.year_for_comp = [2055 2069];
p.ensemble_numbers = [1 2 3];
p.units = '°C';
p.years_of_ss = [2050 2069];
p.wraparound = 0;
% Longitude and latitude
load get_lat_and_lon.mat
if p.wraparound ==1
lon = [lon;360];
end 
ww = cos(lat/180*pi);
p.ww = ww;
p.lat = lat;
p.lon = lon;
p.latbounds = [-inf inf];
p.lonbounds = [-inf inf];
p.latbounds = [-inf inf];
p.lonbounds = [-inf inf];
%% CO2
load Variables_for_multilat_emulator_precipitation.mat
load Variables_for_multilat_emulator_temperature.mat
name_array = ["60°N SAI"; "30°N SAI"; "15°N SAI"; "0°N SAI"; "15°S SAI"; "30°S SAI"; "60°S SAI"; "GHG Warming"];
%%
tiley = tiledlayout(4,2);
% set(gcf, 'Position', [200, 200, 1800,1200]) % Set figure size
set(gcf, 'Position', [200, 200, 1200,1000]) % Set figure size
tiley.Padding = 'tight';
load coastlines

for i = 1:8
    if i<8
        coeff = -1;
    else
        coeff = 1;
    end
    current_T_pattern=coeff*all_T_patterns_scaled(:,:,i);
    current_T_pattern_map = [current_T_pattern;current_T_pattern(1,:)];
    nexttile 
    worldmap('World');
    box on
    hold on
    mmin = -10;
    mmax = 10;
    l_colb = 50;
    key_for_color = linspace(mmin,mmax,l_colb);
    fc = brewermap(l_colb,'*RdBu');
    number_of_levels=6;
    v2 = mmin:(mmax-mmin)/(l_colb/5):mmax;
    hold on
    contourfm(lat,[lon;360],double(current_T_pattern_map'),mmin:(mmax-mmin)/l_colb:mmax,'LineColor','none');
    colormap(fc)
    geoshow(coastlat,coastlon,'Color','k')
    mlabel off; plabel off; gridm off
    hold off
    clim([mmin mmax])
    cbmap1=flipud(cbrewer('div','*RdBu',number_of_levels));
    hl = colorbar('Position',[0.965 0.168 0.011 0.7],'YTick',v2);
    hl.FontSize = 14;
    title(name_array(i) + " Temperature Pattern, °C")
    set(gca,"Fontsize",14)
end
% hl = colorbar(h,'Position',[0.965 0.168 0.011 0.7],'YTick',v2);
% hl.fontsize = 16;

set(gcf,'renderer','painters')
print(gcf,'-dpng',["Uncoord_Emu_Paper/Uncoord_Plots/All_T_Patterns_" + getNow() + ".png"],'-r300')
close all
%%
tiley = tiledlayout(4,2);
% set(gcf, 'Position', [200, 200, 1800,1200]) % Set figure size
set(gcf, 'Position', [200, 200, 1200,1000]) % Set figure size
tiley.Padding = 'tight';
load coastlines

for i = 1:8
    if i<8
        coeff = -1;
    else
        coeff = 1;
    end
    current_P_pattern=coeff*all_P_patterns_scaled(:,:,i);
    current_P_pattern_map = [current_P_pattern;current_P_pattern(1,:)];
    nexttile 
    worldmap('World');
    box on
    hold on
    mmin = -50;
    mmax = 50;
    l_colb = 50;
    key_for_color = linspace(mmin,mmax,l_colb);
    fc = brewermap(l_colb,'*RdBu');
    number_of_levels=6;
    v2 = mmin:(mmax-mmin)/(l_colb/5):mmax;
    hold on
    contourfm(lat,[lon;360],double(current_P_pattern_map'),mmin:(mmax-mmin)/l_colb:mmax,'LineColor','none');
    colormap(fc)
    geoshow(coastlat,coastlon,'Color','k')
    mlabel off; plabel off; gridm off
    hold off
    clim([mmin mmax])
    cbmap1=flipud(cbrewer('div','*RdBu',number_of_levels));
    
    title(name_array(i) + " Preciptiation Pattern, mm/day")
    set(gca,"Fontsize",14)  
    hl = colorbar('Position',[0.965 0.168 0.011 0.7],'YTick',v2);
    hl.FontSize = 14;
end

h = axes(tiley,'Visible','off');
h1 = colorbar(h,'Position',[0.965 0.168 0.011 0.7],'YTick',v2);
% h1.fontsize = 16;

set(gcf,'renderer','painters')
print(gcf,'-dpng',["Uncoord_Emu_Paper/Uncoord_Plots/All_P_Patterns_" + getNow() + ".png"],'-r300')

close all