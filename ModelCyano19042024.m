%% MODEL GENERAL INPUTS
clear
clc
tic
%% OPERATIONAL INPUTS
% System temperature (K)
T = 31.74 + 273.15; % (ºC converted to Kelvin)
% Total regolith inside the system (kg)
regolith_total = 200*0.8;%  [1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]; %[1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200];[200, 180, 160, 140, 120, 100, 80, 60, 40, 20, 1]% Culture volume (m3)
V_culture = 1; % Should be kept at 1 m3
% Height of culture (m)
H_culture = 1.7; 
% Maximum ingoing gas velocity (m s-1)
v_gs_max = 0.08; % (F_G should not go over 0.0083)
% Relative humidity (%) 
RH = 100;
% PAR photon flux density at the surface of the PBR (molph m-2 s-1)
Iph_0 = (6.5*10^-4); %6.5*10^-4;
% Mass fraction of perchlorates in the regolith (wt %)
wt_ClO4 = 0.4;  %0.4; % Mars surface regolith has an average wt % of 0.4 to 0.6 Hecht 2009, but can be lower or higher
% Regolith surface area (m2 kg-1)
BET = 7.6054e+03; 
% Initial biomass concentration (mol m-3)
Cx0 = (0.02/32.975*1000);
% Total cultivation time (Days converted to seconds)
time = 30*24*60*60; 
% Initial phosphorus concentration (mol m-3)
C_P0 = 0;
% Regolith shading - Place 1 to account for regolith shading or 0 otherwise
Regolith_shading = 0; 
% Expected bioprocess operation time (years)
Lifetime = 10;
% BIOLOGICAL INPUTS
% Molecular weight biomass (g molC-1 OR g molx-1)
Mx = 32.975; % (This study)
% Maximum growth rate (s-1)
mu_max = 8.912e-06; % (This study)
% Growth rate from the no-perchlorate control (s-1)
mu_max_ClO4 = (0.435642247)/(24*60*60); % (Ramalho et al., 2022)
% Maximum growth rate from where the half-velocity constants were obtained (s-1)
mu_max_GAS = (1.054)/(24*60*60); % (Verseux et al., 2024)

% Temperature model inputs (Barbera et al., 2019)
% Minimum growth temperature (K)
T_min = 5.29+273.15; % (ºC converted to Kelvin)
% Maximum growth temperature (K)
T_max = 42.93+273.15; % (ºC converted to Kelvin)
% Optimum growth temperature (K)
T_opt = 31.74+273.15; % (ºC converted to Kelvin)
% Biomass growth stoichiometric relationship and respective coefficients (Barbera et al., 2019)
%1.0882 H2O + 0.0873 N2 + CO2 + 0.0078 H2PO4- + 0.0078 HPO4(2-) ->
% (C /H 2.1763/O 0.9551/N 0.1746/P 0.0156) + 1.0860 O2 + 0.0233 OH-  
% O2 Coefficient (molO2 molx-1)
O2_COEFF = 1.0860;
% N2 Coefficient (molN2 molx-1)
N2_COEFF = 0.0873;
% P Coefficient (molP molx-1)
P_COEFF = 0.0036/0.2314;
% Nitrogen half-velocity constant (molN2 m-3)
K_N = 0.032596; % (Verseux et al., 2024)
% Carbon half-velocity constant (molDIC m-3) 
K_DIC = 0.045732; % (Verseux et al., 2024)
% Selectivity coefficient of Rubisco for CO2 in regards to O2 
S = 60; % (Empirical rule of thumb)
% Biomass spectrally averaged specific light absorption coefficient (m2 molx-1)
a_x = 3.3145; % (This study)  
% Regolith spectrally averaged specific light absorption coefficient (m2 kg-1)
a_r = 15.836; % (Ramalho et al., 2022)
% Perchlorate inhibition constant 1 (molClO4- m-3) 
K_ClO4_1 = 1.12380*10^-8; % (Ramalho et al., 2022)
% Perchlorate inhibition constant 2 (molClO4- m-3) 
K_ClO4_2 = 5.70311*10^-8; % (Ramalho et al., 2022)
% Inhibition constant for phosphate (molP m-3)
Ki_P = 1.379; % (Ramalho et al., 2022)
% Phosphorus half velocity constant (molP m-3)
K_P = 0.0149; % (Ramalho et al., 2022) 
% Light half velocity constant (molph m-2 s-1)
K_L = (2.498*10^-5); % (This study)
% CONSTANTS AND MATERIAL INPUTS
% Molecular weight perchlorate (g mol-1) 
Mw_ClO4 = 99.45;  
% Molecular weight oxygen (g mol-1) 
Mw_O2 = 15.999*2;
% Molecular weight nitrogen (g mol-1) 
Mw_N2 = 14.0067*2;
% Molecular weight carbon dioxide (g mol-1) 
Mw_CO2 = 44.01;
%Gas constant (L kPa mol-1 K-1)
R = 8.3144598;  
% Water density (kg m-3)
rho_H2O = 996.8; 
% Regolith density (kg m-3)
rho_reg = 1290; % (Density of Mars regolith simulant)
% Partition coefficient O2 gas over liquid (mol mol-1)
m_O2 = 31.55; % (Henry's law)
% Partition coefficient CO2 gas over liquid (mol mol-1)
m_CO2 = 1.21; % (Henry's law)
% Partition coefficient N2 gas over liquid (mol mol-1)
m_N2 = 63.3141; % (Henry's law)
% Diffusion coefficient for O2 at 25ºC (cm2 s-1)
D_O2 = 2.42*10^-5; 
% Diffusion coefficient for CO2 at 25ºC (cm2 s-1)
D_CO2 = 1.91*10^-5; 
% Diffusion coefficient for N2 at 25ºC (cm2 s-1)
D_N2 = 2*10^-5; 
% Engineering ToolBox, (2008). Gases Solved in Water - Diffusion Coefficients. [online] 
% Available at: https://www.engineeringtoolbox.com/diffusion-coefficients-d_1404.html [Accessed 22 Apr. 2023].
% Earth ambient pressure (mbar)
P_EARTH = 1013.25;
% Martian gravity (m s-2)
g = 3.711; 
% Low gravity gas-liquid transfer correction factor
g_corr = 0.79; % (Pettit and Allen 1992)
% Phosphorus release rate (mol m-2 s-1)
R_P = 3.442E-13; 
% PRELIMINARY CALCULATIONS
% Base radius (m)
Base_r = sqrt(V_culture/(pi*H_culture));
% Checkpoint to ensure the height of the culture is not double the diameter of the photobioreactor.
% This is a rule of thumb to guarantee that the reactor is properly mixed.
% A taller reactor is possible but extra effort should be made to guarantee
% that the culture is homogenous. 
if H_culture/(Base_r*2)<=2 
    else
        disp("H_culture/(Base_r*2)HIGHER THAN 2")
end
% TEMPERATURE MODULE
% Employing the model from Bernard, O. & Rémond, B. 
% Validation of a simple model accounting for light and temperature effect 
% on microalgal growth. Bioresour. Technol. 123, 520–527 (2012)

% Checkpoint for Property 1 of the temperature model
if (T_max+T_min)/2 < T_opt
else
    error('Property 1 of the temperature model [(T_max+T_min)/2 < T_opt] is not satisfied')
end

% TEMPERATURE MODULE: Fraction of maximum growth at defined system temperature (f_T)
% Fraction of maximum growth rate achieved at the defined system temperature (Dimensionless) 
f_T = ((T-T_max).*((T-T_min).^2))./((T_opt-T_min).*((T_opt-T_min).*(T-T_opt)-(T_opt-T_max).*(T_opt+T_min-2.*T))); 
% Correction for negative growth 
f_T = max(f_T,0);

% TEMPERATURE MODULE: Plot
% This plot is only available when working with a range of temperatures!
plot(T,f_T)
ylim([0 1])
xlim([T_min T_max])
xlabel('Temperature (K)');
ylabel('F_T');

% PRODUCTION MODULE: Preliminary calculations for light module
% Light Source/Guide specifications
% If the downcomer area should be twice as great as riser (m)
r_i = Base_r/sqrt(2.7);
% Adjustment of center of PBR (To avoid dividing by zero) (m)
r_adjust = 0.01;
% Light guide specifications; further calcution of radii in m
% If a light guide is added at the centre of the PBR it should be named with
% LG1 and the draft tube light guide is then LG2 (LG2_r=r_i)
LG1_r = r_i + r_adjust;         %Draft tube as light guide 1 m)
LG2_r = Base_r + r_adjust;      %outer Shell as light guide 3 m)
Radius = Base_r + r_adjust;   

 % PRODUCTION MODULE: Identification of optimal regolith concentration and photobioreactor operation strategy
for r = 1:length(regolith_total) 
    % Regolith concentration (kg m-3)
    C_R = regolith_total(r)./V_culture;
    % Concentration of perchlorates in the medium (mol m-3)
    C_ClO4 = (C_R*(wt_ClO4/100))/(Mw_ClO4/1000); 
    % Growth rates as a function of perchlorate concentration (s-1)
    mu_ClO4 = mu_max_ClO4 - K_ClO4_1*C_ClO4^2 - K_ClO4_2*C_ClO4; 
    % Fraction of the maximum growth rate as a function of perchlorate concentration
    f_ClO4 = mu_ClO4/mu_max_ClO4;
    if f_ClO4 <= 0
        disp('f_ClO4 is negative so growth is not possible')
        break
    end
    % OD45 function to calculate concentration of biomass and phosphorus
    % over cultivation time
    [t,y] = ode45(@(t,y) odefcnREGOLITHLIGHTMONOD(t,y,R_P,BET,C_R,mu_max,K_P,P_COEFF,f_ClO4,f_T,a_r,a_x,Iph_0,r_adjust,K_L,Base_r,r_i,LG1_r,LG2_r,Regolith_shading,Ki_P),[1 time],[C_P0 Cx0]); 
    % Define biomass concentration (molx m-3)
    Cx = y(:,2);
    % Define phosphorus concentration (molP m-3)
    C_P = y(:,1);
    % Plot biomass concentration
    figure(1), plot(t,Cx); hold all;
    % Plot phosphorus concentration
    figure(2), plot(t,C_P); hold all;
    % The phosphorus (C_P) and biomass (Cx) concentrations at each time
    % point of the batch ODE serve as input for the initial phosphorus
    % (C_P0) and set biomass (Cx_set) concentrations of the steady state
    % Define time for production phase/ steady state
    time_loop = t(:,1); 
    TIMESIZE = length(t);
    % Loop for production phase 
    for i = 1:TIMESIZE 
        % Set biomass concentration (molx m-3)
        Cx_set = Cx(i);
        % Initial flow of water out for timepoint i (m3 s-1)
        f_w_out0 = -V_culture.*((mu_max.*C_P(i))./(K_P+(C_P(i).*(1+C_P(i)./Ki_P)))).*f_ClO4.*f_T; 
        % Steady state time - Time from batch to total cultivation time (s)
        timeSS(i) = time-time_loop(i);
        % Cultivation time where there is a switch between batch phase and
        % steady state (s)
        timeSWITCH(i) = time_loop(i);
        % Calculation of average specific growth rate as a function of
        % light (mu_L_avg)
                % If loop to define if regolith shading should be
                % considered or not as per input in "Operational inputs
                % section"
                if Regolith_shading == 1 
                    % Regolith attenuation (m-1)
                    Atten_reg = a_r.*C_R;
                else
                    Atten_reg = 0;
                end
                % Biomass attenuation (m-1)
                Atten_x = a_x.*Cx_set;
                % IF section for two light rods (draft tube and outer shell)
                if exist('LG3_r','var') == 0

                    r_center = r_adjust;                %Adjustments to LG1 and LG2

                    r1_RI = linspace(LG1_r,r_center,1000);        %riser path 
                    r1_DC = linspace(LG1_r,LG2_r,1000);    %downcomer path LG1
                    r2_DC = linspace(LG2_r,LG1_r,1000);     %dowcnomer path LG2   

                    %Light Intensities LG1 downcomer (DC = downcomer)
                    CombA_LG1_DC = (LG1_r*Iph_0./r1_DC).*exp((-Atten_reg-Atten_x)*(r1_DC-LG1_r));    %Both attenutation effects combined
                    %Light Intensities LG1 Riser (RI = Riser)
                    CombA_LG1_RI = (LG1_r*Iph_0./r1_RI).*exp((-Atten_reg-Atten_x)*(LG1_r-r1_RI));
                    %Light intensities LG2 downcomer 
                    CombA_LG2_DC = (LG2_r*Iph_0./r2_DC).*exp((-Atten_reg-Atten_x).*(LG2_r-r2_DC));                   
                    %Combining values of overlapping curves for downcomer
                    Iph_DC = CombA_LG1_DC + fliplr(CombA_LG2_DC);   
                    %Light Intensity Riser
                    Iph_RI = CombA_LG1_RI;
                    %Putting the path length and light flux in one variable
                    Iph_RIPath = [r1_RI;Iph_RI];
                    % Radius of draft tube (LG1_r) becomes zero for Downcomer path
                    % Definition of flux (Iph) with new zero point
                    Iph_DCPath= [r1_DC-r1_DC(1);Iph_DC];

                % Calculation for a given location of LG (LG3_r) 
                else 
                    msgbox("The separate Light Module for three light rods has to be used!");
                end

               
                % Monod calculation of growth rate as a function of light
                % intensity 
                % All growth rates are in (s-1)
                % Growth rate vector in riser
                mu_L_RI = mu_max.*(Iph_RI./(K_L+Iph_RI));
                % Average growth rate in riser
                mu_L_RI_avg = mean(mu_L_RI);
                % Growth rate vector in downcomer 
                mu_L_DC = mu_max.*(Iph_DC./(K_L+Iph_DC));
                % Average growth rate in downcomer
                mu_L_DC_avg = mean(mu_L_DC);
                % Weighted average growth rate in photobioreactor
                mu_L_avg_SS = (r_i/Base_r).*mu_L_RI_avg + ((Base_r-r_i)/Base_r).*mu_L_DC_avg;
       
        if timeSWITCH(i) == time
            % Biomass produced is equal to final biomass concentration during batch when
            % there is no steady state (mol x)
            Biomass_produced = Cx_set*V_culture;
        else
            % ODE45 function to calculate phosphorus concentration and reactor dilutions over the production phase / steady state
            [a,C_P_SS] =ode45(@(a,b) odefcnSTEADYSTATEREGOLITHLIGHTMONOD(a,b,R_P,BET,C_R,mu_max,K_P,P_COEFF,f_ClO4,V_culture,Cx_set,f_T,mu_L_avg_SS,Regolith_shading,Ki_P,C_P0),[timeSWITCH(i) time],C_P(i));
            % Flow of water (m3 s-1)
            f_w_out = -V_culture.*min(((mu_max.*C_P_SS)./(K_P+(C_P_SS.*(1+C_P_SS./Ki_P)))),mu_L_avg_SS).*f_ClO4.*f_T; 
            % Total water out (m3)
            w_out = trapz(a,f_w_out); 
            % Total biomass produced (molx)
            Biomass_produced = -w_out*Cx_set + Cx_set*V_culture;
        end
        % Record biomass produced for batch to production phase switch in timepoint i (molx)
        Biomass(i) = Biomass_produced;
        % Record maximum biomass for a given regolith concentration r
        % (molx)
        OPT_point_x(r) = max(Biomass);
        OPT_point_x_kg = OPT_point_x*Mx/1000
        % Record switch time in which the biomass production was maximum for a
        % given regolith concentration r (s)
        OPT_point_t(r) = timeSWITCH(Biomass == max(Biomass));
        %Batch_over_production = OPT_point_t/time
        %Biomass_over_time = OPT_point_x_kg/(time/(24*60*60))
    end
     REGOLITH(r) = C_R;
end
% Maximum biomass produced (molx)
Maximum_biomass_produced = max(OPT_point_x);
% Regolith concentration corresponding to the maximum biomass production
% (kg m-3)
Maximum_biomass_produced_Regolith = REGOLITH(OPT_point_x == max(OPT_point_x));
% Switch time corresponding to the maximum biomass production(s)
Maximum_biomass_produced_Switch = OPT_point_t(OPT_point_x == max(OPT_point_x));

%% PRODUCTION MODULE: Run batch phase with optimized regolith concentration and photobioreactor operation
% Redefine regolith concentration according to optimum value (kg m-3)
C_R = Maximum_biomass_produced_Regolith; 
% Rerun batch phase to record values at the given regolith concentration
% Concentration of perchlorates in the medium (mol m-3)
C_ClO4 = (C_R*(wt_ClO4/100))/(Mw_ClO4/1000); 
% Growth rates as a function of perchlorate concentration (experimental) (s-1)
mu_ClO4 = mu_max_ClO4 - K_ClO4_1*C_ClO4^2 - K_ClO4_2*C_ClO4;
% Fraction of the maximum growth rate as a function of perchlorate concentration
f_ClO4 = mu_ClO4/mu_max_ClO4;
if f_ClO4 <= 0
    disp('f_ClO4 is negative so growth is not possible') 
    return
end
% Batch OD45 function to calculate concentration of biomass and phosphorus over cultivation time - Now only runs until the time switch optimized for biomass production
[t,y] = ode45(@(t,y) odefcnREGOLITHLIGHTMONOD(t,y,R_P,BET,C_R,mu_max,K_P,P_COEFF,f_ClO4,f_T,a_r,a_x,Iph_0,r_adjust,K_L,Base_r,r_i,LG1_r,LG2_r,Regolith_shading,Ki_P),[1 Maximum_biomass_produced_Switch],[C_P0 Cx0]); 
% Define biomass concentration (molx m-3)
Cx = y(:,2);
% Define phosphorus concentration (molP m-3)
C_P = y(:,1);
% Checkpoint to ensure the concentration of phosphorus does not reach
% irrealistic values
if C_P >= 0.5
    disp('The concentration of dissolved phosphorus might be over the saturation limit')
end
% If you want to plot biomass or phosphorus concentrations over the batch
% phase just activate the two following plot functions
% figure(1), plot(t,Cx); 
% figure(2), plot(t,C_P); 
% Record growth rates over the batch phase (s-1)
for m = 1:length(Cx) 
    if m == 1
        % Growth rate during batch (s-1)
        mu_batch = (log(Cx(m)/(Cx0)))/t(m);
        % Biomass productivity during batch (molx m-3 s-1)
        x_r_batch = mu_batch.*(Cx(m));
    else
        % Growth rate during batch (s-1)
        mu_batch = log(Cx(m)/Cx(m-1))/(t(m)-t(m-1));
        % Biomass productivity during batch (molx m-3 s-1)
        x_r_batch = mu_batch.*(Cx(m));
    end
    % Growth rate vector during batch (s-1)
    mu_batch_loop(m) = mu_batch;
    % Biomass productivity vector during batch (molx m-3 s-1)
    x_r_batch_loop(m) = x_r_batch;
end

% Oxygen production over the batch phase (mol)
dO2_batchdt = (Cx-Cx0)*O2_COEFF;
% Carbon dioxide consumption over the batch phase (mol)
dCO2_batch_dt = (Cx-Cx0);
% Nitrogen consumption over the batch phase (mol)
dN2_batch_dt = (Cx-Cx0).*N2_COEFF;
% Record biomass concentration over time in batch phase 
Cx_batch = Cx; 
% Total oxygen production in batch phase (mol)
O2_batch_mol = ((Cx_set-Cx0)*O2_COEFF).*V_culture;
% Carbon dioxide consumption in batch phase (mol)
CO2_batch_mol = ((Cx_set-Cx0)).*V_culture;
% Nitrogen consumption in batch phase (mol)
N2_batch_mol = ((Cx_set-Cx0).*N2_COEFF).*V_culture;

%% PRODUCTION MODULE: Plot batch phase
plot(t,Cx_batch)
xlabel('t (s^-^1)');
ylabel('Biomass concentration (mol_x m^-^3)');
xlim([0 Maximum_biomass_produced_Switch])

%% PRODUCTION MODULE: Run production phase with optimized regolith concentration and photobioreactor operation
% Define set biomass concentration as the final biomass from the batch
% phase (molx m-3)
Cx_set = max(Cx);
% Define set initial phosphorus concentration as the final  from the batch
% phase (molP m-3)
C_P_SS_0 = C_P(end);
% Initial flow of water out at the defined optimum switch point (m3 s-1)
f_w_out0 = -V_culture.*((mu_max.*C_P_SS_0)./(K_P+(C_P_SS_0.*(1+C_P_SS_0./Ki_P)))).*f_ClO4.*f_T; 
% Steady state time - Time from batch to total cultivation time (s)
timeSS = time-Maximum_biomass_produced_Switch;
% Calculation of mu_L_avg
% If loop to define if regolith shading should be
% considered or not as per input in "Operational inputs section"
if Regolith_shading == 1 
    % Regolith attenuation (m-1)
    Atten_reg = a_r.*C_R;
    else
    Atten_reg = 0;
end
% Calculation of biomass attenuation coefficient (m-1)
Atten_x = a_x.*Cx_set;
% IF section for two light rods (draft tube and outer shell)
if exist('LG3_r','var') == 0
    
    r_center = r_adjust;                %Adjustments to LG1 and LG2
    
    r1_RI = linspace(LG1_r,r_center,1000);        %riser path 
    r1_DC = linspace(LG1_r,LG2_r,1000);    %downcomer path LG1
    r2_DC = linspace(LG2_r,LG1_r,1000);     %dowcnomer path LG2   

    %Light Intensities LG1 downcomer (DC = downcomer)
    CombA_LG1_DC = (LG1_r*Iph_0./r1_DC).*exp((-Atten_reg-Atten_x)*(r1_DC-LG1_r));    %Both attenutation effects combined
    %Light Intensities LG1 Riser (RI = Riser)
    CombA_LG1_RI = (LG1_r*Iph_0./r1_RI).*exp((-Atten_reg-Atten_x)*(LG1_r-r1_RI));
    %Light intensities LG2 downcomer 
    CombA_LG2_DC = (LG2_r*Iph_0./r2_DC).*exp((-Atten_reg-Atten_x).*(LG2_r-r2_DC));                   
    %Combining values of overlapping curves for downcomer
    Iph_DC = CombA_LG1_DC + fliplr(CombA_LG2_DC);   
    %Light Intensity Riser
    Iph_RI = CombA_LG1_RI;
    %Putting the path length and light flux in one variable
    Iph_RIPath = [r1_RI;Iph_RI];
    % Radius of draft tube (LG1_r) becomes zero for Downcomer path
    % Definition of flux (Iph) with new zero point
    Iph_DCPath= [r1_DC-r1_DC(1);Iph_DC];

% Calculation for a given location of LG (LG3_r) 
else 
    msgbox("The separate Light Module for three light rods has to be used!");
end

% Monod calculation of growth rate as a function of light intensity 
% All growth rates are in (s-1)
% Growth rate vector in riser
mu_L_RI = mu_max.*(Iph_RI./(K_L+Iph_RI));
% Average growth rate in riser
mu_L_RI_avg = mean(mu_L_RI);
% Growth rate vector in downcomer 
mu_L_DC = mu_max.*(Iph_DC./(K_L+Iph_DC));
% Average growth rate in downcomer
mu_L_DC_avg = mean(mu_L_DC);
% Weighted average of light-depenedent growth rate in photobioreactor
% during continuous production/steady state
mu_L_avg_SS = (r_i/Base_r).*mu_L_RI_avg + ((Base_r-r_i)/Base_r).*mu_L_DC_avg;

if Maximum_biomass_produced_Switch == time
    % Biomass produced is equal to final biomass concentration during batch when
    % there is no steady state (mol x)
    Biomass_produced = Cx_set;
else
    % ODE45 function to calculate phosphorus concentration and reactor
    % dilutions over the production phase / steady state - Now using values
    % from optimized run
    [a,C_P_SS] =ode45(@(a,b) odefcnSTEADYSTATEREGOLITHLIGHTMONOD(a,b,R_P,BET,C_R,mu_max,K_P,P_COEFF,f_ClO4,V_culture,Cx_set,f_T,mu_L_avg_SS,Regolith_shading,Ki_P,C_P0),[Maximum_biomass_produced_Switch time],C_P_SS_0);
    % Flow of water (m3 s-1)
    f_w_out= -V_culture.*min(((mu_max.*C_P_SS)./(K_P+(C_P_SS.*(1+C_P_SS./Ki_P)))),mu_L_avg_SS).*f_ClO4.*f_T; 
    % Growth rate over the production phase / steady state (s-1)
    mu_SS = min(((mu_max.*C_P_SS)./(K_P+(C_P_SS.*(1+C_P_SS./Ki_P)))),mu_L_avg_SS).*f_ClO4.*f_T;
    % Biomass productivity over the production phase / steady state (molx m-3 s-1)
    x_r_SS = Cx_set.*mu_SS;
    % Total water out (m3)
    w_out = trapz(a,f_w_out); 
    % Total biomass produced (molx)
    Biomass_produced = -w_out*Cx_set + Cx_set.*V_culture;
end
% Checkpoint to ensure the concentration of phosphorus does not reach
% irrealistic values 
if C_P_SS >= 0.5
    disp('The concentration of dissolved phosphorus might be over the saturation limit - double-check')
end
% Total biomass produced (kg)
Biomass_kg = (Biomass_produced*Mx)./1000;
% Mass balances
% Biomass mass balance
%0 == x_r.*V_culture + Cx_in.*f_w_in + f_w_out*Cx; 
% Water mass balance
%0 == f_w_in + f_w_out
% Flow of water in (m3 s-1)
f_w_in = - f_w_out; 
% Total water input (m3)
Water_m3 = -w_out + V_culture;
% Total water input (kg)
Water_kg = Water_m3*1000;
% Regolith mass balance
%0 == f_w_out*C_R + f_reg_in; 
% Flow of regolith in (kg s-1)
f_reg_in = -f_w_out*C_R; 
% Flow of regolith out (kg s-1)
f_reg_out = -f_reg_in;
% Total regolith input (kg)
Regolith_kg = -w_out.*C_R + C_R;
% Total Oxygen produced (mol)
Oxygen_mol = Biomass_produced.*O2_COEFF;
% Total Oxygen produced (kg)
Oxygen_kg = (Oxygen_mol.*Mw_O2)./1000;
% Total Carbon dioxide consumed (mol)
CarbonDioxide_mol = Biomass_produced;
% Total Carbon dioxide consumed (kg)
CarbonDioxide_kg = (CarbonDioxide_mol.*Mw_CO2)./1000;
% Total Nitrogen consumed (kg)
Nitrogen_mol = Biomass_produced.*N2_COEFF;
% Total Nitrogen consumed (mol)
Nitrogen_kg = (Nitrogen_mol.*Mw_N2)./1000;

%% GAS 1 MODULE: Preliminary calculations
% Reactor cross sectional area (m2)
Reactor_Area = pi.*(Base_r.^2);
% Medium density (kg m-3)
rho_M = 1/((C_R/((C_R+rho_H2O*10^3)*rho_reg))+ (rho_H2O*10^3/((C_R+rho_H2O*10^3)*rho_H2O)));

%% GAS 1 MODULE: Minimal concentrations of DIC and N2_L 
% Growth rates over the entire cultivation length (s-1) 
if Maximum_biomass_produced_Switch == time
    % Growth rate (s-1)
    mu_batch_SS = mu_batch_loop;
    % Flow of water out (m3 s-1)
    f_w_out = 0; 
    % Flow of water in (m3 s-1)
    f_w_in = 0; 
else
    % Growth rate (s-1)
    mu_batch_SS = [mu_batch_loop,mu_SS']; % Note that this is not representative of time - the number of points is dependent on the ODE 
    % Flow of water out (m3 s-1)
    f_w_out = [mu_batch_loop*0,-mu_SS'.*V_culture]; 
    % Flow of water in (m3 s-1)
    f_w_in = -f_w_out; 
end

% If the code is taking too long to run here you may need to cut the vector
% down. You can just activate the following code to reduce the vector by a
% factor of 10
mu_batch_SS = mu_batch_SS(:,1:10:end);
f_w_in = f_w_in(:,1:10:end);
f_w_out = f_w_out(:,1:10:end);

% Define minimal concentrations of DIC and N2_L over the cultivation period
for c = 1:length(mu_batch_SS)
    % Define growth rate (s-1)
    mu = mu_batch_SS(c);
    % Define DIC-dependent growth rate (s-1)
    mu_DIC = mu; 
    % DIC concentration (mol m-3)
    syms C_DIC
    C_DIC = double(solve(mu_DIC == mu_max_GAS.*(C_DIC./(C_DIC+K_DIC)),C_DIC));
    % Record DIC value
    C_DIC_loop(c) = C_DIC;
    % Define N2-dependent growth rate (mol m-3)
    mu_N = mu; 
    % Liquid N2 concentration (mol m-3)
    syms C_N2_L
    C_N2_L = double(solve(mu_N == mu_max_GAS.*(C_N2_L./(C_N2_L+K_N)),C_N2_L));
    % Record N2_L value
    C_N2_L_loop(c) = C_N2_L;
end

%% GAS 1 MODULE: Carbon and Nitrogen transfer rates
% Define ingoing water as pure i.e. no dissolved gases 
% Dissolved oxygen concentration in ingoing water (molO2 m-3)
C_O2_L_in = 0;
% Dissolved carbon dioxide concentration (molCO2 m-3)
C_CO2_L_in = 0;
% Dissolved nitrogen concentration (molN2 m-3)
C_N2_L_in = 0;
% Define biomass productivity (molx m-3 s-1)
if Maximum_biomass_produced_Switch == time
    % Biomass productivity (molx m-3 s-1)
    x_r = x_r_batch_loop;
    % If the code is taking too long to run here you may need to cut the vector
    %  down. You can just activate the following code to reduce the vector by a
    % factor of 10
    x_r = x_r(:,1:10:end);
    % Store biomass concentration over time for light 2 module (molx m-3)
    Cx_light2 = x_r./mu_batch_SS;
else
    % Biomass productivity (molx m-3 s-1)
    x_r = [x_r_batch_loop,x_r_SS'];
    % If the code is taking too long to run here you may need to cut the vector
    %  down. You can just activate the following code to reduce the vector by a
    % factor of 10
    x_r = x_r(:,1:10:end);
    % Store biomass concentration over time for light 2 module (molx m-3)
    Cx_light2 = x_r./mu_batch_SS;
end

% Concentration of dissolved carbon dioxide (molCO2 m-3)
C_CO2_L = C_DIC; % For this analysis DIC and CO2_L are assumed to be interchangeable
% Oxygen productivity (molO2 m-3 s-1)
O2_r_preloop = x_r.*O2_COEFF; 
% CO2_aq/HCO3 consumption (molCO2 m-3 s-1) 
CO2_r = -x_r; 
% N2 consumption (molN2 m-3 s-1)
N2_r = x_r.*(-N2_COEFF);
% Outgoing flow is assumed to have the same characteristics of the medium
C_CO2_L_out = C_CO2_L;  
% Carbon dioxide transfer rates (molCO2 m-3 s-1)
CTR_preloop = -(f_w_in.*C_CO2_L_in + f_w_out.*C_CO2_L_out + CO2_r.*V_culture)./V_culture; 
% Outgoing flow is assumed to have the same characteristics of the medium 
C_N2_L_out = C_N2_L;
% Nitrogen transfer rates (molN2 m-3 s-1)
NTR_preloop = -(f_w_in.*C_N2_L_in + f_w_out.*C_N2_L_out + N2_r.*V_culture)./V_culture; 
%% GAS 1 MODULE: Water vapour pressure
% Water saturation pressure (mbar)
P_H2O_S_AMB = exp(-6096.9385.*(T^-1)+21.2409642-2.711193.*10^-2.*T+1.673952.*10^-5.*(T^2)+2.433502.*log(T)).*0.01; %(Sonntag, 1990)
% Dew point temperature (K)
t_dp = (((243.12.*log(P_H2O_S_AMB.*100./611.2))./(17.62-log(P_H2O_S_AMB.*100./611.2))))+273.15; %(BS1339-1, 2011)
% Vapour pressure at ambient pressure (mbar)
P_H2O_AMB = (RH.*P_H2O_S_AMB)./100; %(BS1339-1, 2008)
% Correction factor for non-pure gas mixtures - Air (Dimensionless)
f_w_P_tdp = 1 + (10^-6.*(P_H2O_AMB.*100)./(t_dp)).*((38+173.*exp(-t_dp/43)).*((1-((P_H2O_AMB.*100)./(P_EARTH.*100)))+(6.39+4.28.*exp(-t_dp./107)).*(((P_EARTH.*100)./(P_H2O_AMB.*100))-1))); %(BS1339-1, 2008)
P_H2O_AMB = f_w_P_tdp*P_H2O_AMB;

%% GAS 1 MODULE: Total system pressure 

for b = 1:length(x_r)
    % Define dissolved gas concentrations within loop (mol m-3)
    % Dissolved inorganic carbon 
    C_DIC = C_DIC_loop(b);
    % Aqueous carbon dioxide 
    C_CO2_L = C_DIC;
    % Aqueous nitrogen 
    C_N2_L = C_N2_L_loop(b);
    % Define gas transfer rates within loop (mol m-3 s-1)
    % Carbon transfer rate
    CTR = CTR_preloop(b);
    % Nitrogen transfer rate
    NTR = NTR_preloop(b);
    % Oxygen transfer rate
    O2_r = O2_r_preloop(b);
    % Define in and outgoing water flows 
    if Maximum_biomass_produced_Switch == time
        % Flow of water out (m3 s-1)
        f_w_in_insideloop = f_w_in;
        % Flow of water in (m3 s-1)
        f_w_out_insideloop = f_w_out; 
    else
        % Define flow rates within loop (m3 s-1)
        f_w_in_insideloop = f_w_in(b);
        f_w_out_insideloop = f_w_out(b);
    end
    % Find the minimal value of C_O2_L for the pressure of O2 to be positive
    for C_O2_L_Loop = linspace(1,100,10000)
        % Define aqueous concentration of oxygen (molO2 m-3)
        C_O2_L = C_O2_L_Loop./10000;
        % Outgoing flow is assumed to have the same characteristics of the medium
        C_O2_L_out = C_O2_L;
        % Oxygen transfer rate (molO2 m-3 s-1)
        OTR = -(f_w_in_insideloop.*C_O2_L_in + f_w_out_insideloop.*C_O2_L_out + O2_r.*V_culture)./V_culture;
        syms P_gas P_CO2 P_N2 P_O2 P_gas_R P_H2O P_corr
        % Total pressure of the reactor (mbar) - NOTE - the P_N2*0.436
        % term was added to account for the inert gases introduced together
        % with the N2
        eq1 = P_gas == P_CO2 + P_N2 + P_N2*0.436 + P_O2 + P_H2O;
        % Partial pressure of carbon dioxide (mbar)
        eq2 = P_CO2 == ((((((CTR./(sqrt(D_CO2./D_O2).*g_corr.*P_corr.*(0.32.*(v_gs_max.*(P_gas./P_gas_R)).^0.7)))+C_CO2_L).*m_CO2)./1000).*R.*T).*10);
        % Partial pressure of nitrogen (mbar)
        eq3 = P_N2 == ((((((NTR./(sqrt(D_N2/D_O2).*g_corr.*P_corr.*(0.32.*(v_gs_max.*(P_gas./P_gas_R)).^0.7)))+C_N2_L).*m_N2)./1000).*R.*T).*10);
        % Partial pressure of oxygen (mbar)
        eq4 = P_O2 == ((((((OTR./(g_corr.*P_corr.*(0.32.*(v_gs_max.*(P_gas./P_gas_R)).^0.7)))+C_O2_L).*m_O2)./1000).*R.*T).*10);
        % Water vapour pressure (mbar)
        eq5 = P_H2O == P_H2O_AMB; % There is no variation variation of water vapour pressure with total pressure 
        % Total pressure in the water column of the photobioreactor - the division
        % here is to go from Pa to mbar 
        eq6 = P_gas_R == (P_gas + (0.5.*g.*rho_M.*H_culture)/100);
        % Low pressure diffusivity correction
        eq7 = P_corr == sqrt(P_EARTH./P_gas_R); 
        % Define system of equations
        eqns = [eq1 eq2 eq3 eq4 eq5 eq6 eq7];
        % Solve system of equations
        [P_gas, P_CO2, P_N2, P_O2, P_gas_R, P_H2O, P_corr] = vpasolve(eqns, [P_gas P_CO2 P_N2 P_O2 P_gas_R P_H2O P_corr]);
        % Oxygen gas-liquid mass transfer coefficient (s-1)
        k_O2_L_a = P_corr.*g_corr.*(0.32.*(v_gs_max.*(P_gas./P_gas_R)).^0.7);
        % Gaseous concentration of oxygen inside the reactor (molO2 m-3)
        C_O2_G = ((OTR./k_O2_L_a)+C_O2_L).*m_O2;
        % Maximum gas flow (m3 s-1)
        f_G_max = v_gs_max.*Reactor_Area.*(P_gas_R/P_gas);
        % Define gas flow as maximum (m3 s-1)
        f_G_in = f_G_max;
        % Define outgoing flow as opposite of ingoing flow (m3 s-1)
        f_G_out = - f_G_in;
        % Gaseous concentration of oxygen going into the reactor (molO2 m-3)
        C_O2_G_in = ((OTR.*V_culture-f_G_out.*C_O2_G)./f_G_in).*(P_gas/P_gas_R);
        % Partial pressure of oxygen inside and going out of the reactor
        % (mbar) - the 1000 converts m3 to L and the 10 kPa to mbar
        P_O2_in = ((C_O2_G_in./1000).*R.*T).*10;
        % Ensure calculated ppO2 is positive and as close to zero as possible
        if P_O2_in >= 0
            min_P_O2_in = P_O2_in;
            min_C_O2_L = C_O2_L;
            break
        end
    end
    % Record values for variables within loop
    P_gas_loop(b) = P_gas;
    P_CO2_loop(b) = P_CO2;
    P_N2_loop(b) = P_N2;
    P_O2_loop(b) = P_O2;
    P_gas_R_loop(b) = P_gas_R;
    k_O2_L_a_loop(b) = k_O2_L_a;
    OTR_loop(b) = OTR;
    C_O2_L_loop(b) = C_O2_L;
    CTR_loop(b) = CTR;
    NTR_loop(b) = NTR;
    f_G_out_loop(b) = f_G_out;
    f_G_in_loop(b) = f_G_in;
    P_corr_loop(b) = P_corr;
    %Checkpoint to ensure the for loop includes an acceptable dissolved oxygen concentration
    if P_O2_in < 0
        error('Your range of oxygen concentrations does not include the right C_O2_L')
    end
end

%% GAS 1 MODULE: Gas concentration and individual/total pressures
% Carbon dioxide gas-liquid mass transfer coefficient (s-1)
k_CO2_L_a = sqrt(D_CO2./D_O2).*k_O2_L_a_loop;
% Nitrogen gas-liquid mass transfer coefficient (s-1)
k_N2_L_a = sqrt(D_N2/D_O2).*k_O2_L_a_loop;
% Gaseous concentration of oxygen going out of the reactor (molO2 m-3)
C_O2_G_out = ((OTR_loop./k_O2_L_a_loop)+C_O2_L_loop).*m_O2;
% Gaseous concentration of oxygen inside of the reactor (molO2 m-3)
C_O2_G = C_O2_G_out.*(P_gas_R_loop./P_gas_loop);
% Gaseous concentration of carbon dioxide going out of the reactor (molCO2 m-3)
C_CO2_G_out = ((CTR_loop./k_CO2_L_a)+C_DIC_loop).*m_CO2;
% Gaseous concentration of carbon dioxide inside of the reactor (molCO2 m-3)
C_CO2_G = C_CO2_G.*(P_gas_R_loop./P_gas_loop);
% Gaseous concentration of nitrogen going out of the reactor (molN2 m-3)
C_N2_G_out = ((NTR_loop./k_N2_L_a)+C_N2_L_loop).*m_N2;
% Gaseous concentration of nitrogen inside of the reactor (molN2 m-3)
C_N2_G = C_N2_G.*(P_gas_R_loop./P_gas_loop);
% Partial pressure of oxygen inside and going out of the reactor (mbar)
P_O2_out = ((C_O2_G./1000).*R.*T).*10;
P_O2 = P_O2_out.*(P_gas_R_loop./P_gas_loop);
% Partial pressure of carbon dioxide inside (P_CO2) and going out (P_CO2_out) of the reactor (mbar)
P_CO2_out = ((C_CO2_G./1000).*R.*T).*10;
P_CO2 = P_CO2_out.*(P_gas_R_loop./P_gas_loop);
% Partial pressure of nitrogen inside (P_N2) and going out (P_N2_out) of the reactor (mbar)
P_N2_out = ((C_N2_G./1000).*R.*T).*10;
P_N2 = P_N2_out.*(P_gas_R_loop./P_gas_loop);
% Gaseous concentration of oxygen going into the reactor (molO2 m-3)
C_O2_G_in = (OTR_loop.*V_culture-f_G_out_loop.*C_O2_G_out)./f_G_in_loop;
% Gaseous concentration of carbon dioxide going into the reactor (molCO2 m-3)
C_CO2_G_in = (CTR_loop.*V_culture-f_G_out_loop.*C_CO2_G_out)./f_G_in_loop;
% Gaseous concentration of nitrogen going into the reactor (molN2 m-3)
C_N2_G_in = (NTR_loop.*V_culture-f_G_out_loop.*C_N2_G_out)./f_G_in_loop;
% Partial pressure of oxygen going into the reactor (mbar)
P_O2_in = ((C_O2_G_in./1000).*R.*T).*10;
% Checkpoint to assess if partial oxygen pressure is within the considered range
if double(P_O2_in) < 0
    error('P_O2_in is negative')
end
if  double(P_O2_in) > 1
    error('P_O2_in too high')
end
% Partial pressure of carbon dioxide going into the reactor (mbar)
P_CO2_in = ((C_CO2_G_in./1000).*R.*T).*10;
% Partial pressure of nitrogen going into the reactor (mbar)
P_N2_in = ((C_N2_G_in./1000).*R.*T).*10;
% Total pressure outgoing from the photobioreactor (mbar) 
P_TOTAL_out = P_H2O + P_CO2_out + P_N2_out + P_N2_out*0.436 + P_O2_out;
% Total pressure inside the photobioreactor (halfway)(mbar)
P_TOTAL_R = P_TOTAL_out + (0.5.*g.*rho_M.*H_culture)/100;
% Total pressure inside the photobioreactor (bottom)(mbar)
P_TOTAL_R_bottom = P_TOTAL_out + (g.*rho_M.*H_culture)/100;
% Total pressure going into the photobioreactor (mbar)
P_TOTAL_in = P_H2O + P_CO2_in + P_N2_in + P_N2_in*0.436 + P_O2_in;
% Dissolved oxygen considerations
% Checkpoint to assess whether dissolved oxygen levels may be too high 
R_oxy_carb = (1/S)*(C_O2_L_loop./C_DIC_loop);
if R_oxy_carb >=0.1
    error('The system has significant dissolved oxygen. This may be limiting.')
end
% Shear rate considerations
% Minimum bubble diameter possible for a lower shear rate than 60 s-1 (m)
min_db = (2*v_gs_max)/60; 
% Checkpoint to assess whether there might be issues with shear rates
if f_G_max > 0.0083
    disp('You should check the shear rate - it should not be over 60 s-1')
end
%% GAS 2 MODULE
% Define the point where the reactor pressure is maximal
m = find(P_gas_R_loop == max(P_gas_R_loop));
% Oxygen productivity (molO2 m-3 s-1)
O2_r_preloop = x_r.*O2_COEFF; 
% Oxygen transfer rates (molO2 m-3 s-1)
OTR_Gas2 = O2_r_preloop;
% Oxygen gas-liquid mass transfer coefficient (s-1)
k_O2_L_a_Gas2 = OTR_loop./((C_O2_G./m_O2)-C_O2_L_loop);
% Make any gas-liquid mass transfer coefficient into zero 
k_O2_L_a_Gas2(isnan(k_O2_L_a_Gas2))= 0;
% Loop to determine the lowest v_gs necessary without affecting
% biomass productivity at maximum pressure
for v = 1:length(k_O2_L_a_Gas2)
syms v_gs
v_gs_solver = solve(k_O2_L_a_Gas2(v) == double(P_corr_loop(m))*g_corr.*0.32.*(v_gs.*(P_gas_loop(m)/P_gas_R_loop(m))).^0.7,v_gs);
v_gs_loop(v) = v_gs_solver;
end

%% LIGHT 2 MODULE
% Biomass concentration over cultivation to input into the light 2 module
% (molx m-3)
Cx_light2 = x_r./mu_batch_SS; 
% Substituting the first value with the initial biomass concentration (this
% is necessary as the growth rate at the beginning is 0 so it is a NaN 
Cx_light2(1,1) = Cx0;
% Create a light intensity vector to test
Iph_vector = linspace(0,Iph_0,10000);

for e = 1:length(mu_batch_SS) 
    % Growth rate for timepoint corresponding to position e of vector (s-1)
    mu_batch_SS_loop = mu_batch_SS(e);
    % Biomass concentration for timepoint corresponding to position e of vector (molx m-3)
    Cx_light2_loop = Cx_light2(e);
    % Biomass-mediated light attenuation for timepoint corresponding to position e of vector (molx m-3)
    Atten_x_loop = Cx_light2_loop*a_x;
    for d = 1:length(Iph_vector)
        % Select a light intensity from the light intensity vector
        Iph_0_loop = Iph_vector(d);
        % Calculate the growth rate for that light intensity (s-1)
        mu_batch_SS_loop_vector = (r_i/Base_r).*mean(mu_max.*(((LG1_r.*Iph_0_loop./r1_RI).*exp((-Atten_reg-Atten_x_loop).*(LG1_r-r1_RI)))./(K_L+((LG1_r*Iph_0_loop./r1_RI).*exp((-Atten_reg-Atten_x_loop).*(LG1_r-r1_RI)))))) + ((Base_r-r_i)./Base_r).*mean(mu_max.*(((LG1_r.*Iph_0_loop./r1_DC).*exp((-Atten_reg-Atten_x_loop).*(r1_DC-LG1_r)) + fliplr((LG2_r.*Iph_0_loop./r2_DC).*exp((-Atten_reg-Atten_x_loop).*(LG2_r-r2_DC))))./(K_L+((LG1_r.*Iph_0_loop./r1_DC).*exp((-Atten_reg-Atten_x_loop).*(r1_DC-LG1_r)) + fliplr((LG2_r.*Iph_0_loop./r2_DC).*exp((-Atten_reg-Atten_x_loop).*(LG2_r-r2_DC)))))));
        % Store growth rate value in a loop 
        mu_batch_SS_loop_vector_store(d) = mu_batch_SS_loop_vector;
    end
    % Find out which growth rate value from the for loop is the closest
    % match to the actually calculated growth rate
    [difference_of_closest_match,vector_position_of_closest_match] = min(abs(mu_batch_SS_loop-mu_batch_SS_loop_vector_store));
    % Store the light intensity corresponding to the growth rate closest
    % match in a vector 
    Iph_0_light2(e) = Iph_vector(vector_position_of_closest_match);
end
%% EQUIVALENT SYSTEM MASS MODULE

% Harmonization of time between cultivation modules and cost module
Time_harmonization = Lifetime*365*24*60*60/time;

% Payload cost
% Rocket payload capacity (kg)
Rocket_payload_capacity = 4020;
% Launch cost (USD)
Launch_cost = 62000000;
% Cost per kg (USD/kg)
kg_eq = Launch_cost./Rocket_payload_capacity;

% Energy cost
% Cost per kWe (kg/kWe) Lower - 54; Nominal - 87; Upper 338; NASA's Life
% support Baseline Values and Assumptions Document
P_eq = 87; 

% Thermal control cost
% Cost per kWth (kg/kWth) Nominal - 146; Upper 170; NASA's Life
% support Baseline Values and Assumptions Document
C_eq = 146; 
% To see how much we would actually need to cool perhaps I can use this document? -> https://digitalcommons.calpoly.edu/cgi/viewcontent.cgi?article=1061&context=theses

% Volume cost
% Cost per unshielded volume on the surface of Mars (kg/m3) Nominal 9.16; Upper 13.40; NASA's Life
% support Baseline Values and Assumptions Document 
V_eq = 9.16; 
% Cost per volume of PBR (kg)
Volume_cost = V_culture.*V_eq;

% Crewtime cost
% Crewtime cost on the surface of Mars (kg/CM-h) Lower 0.506; Nominal 0.940; NASA's Life
% support Baseline Values and Assumptions Document 
CT_eq = 0.940;
% Crewtime spent on the cyanobacterial biomass production system (CM-h)
% Assumed 5 hours per year for maintenance - control is done from Earth and
% production is automated
CT_PBR = 5;
% Cost of crewtime
Crew_time_cost = CT_eq.*CT_PBR.*Lifetime;

% Water cost
% Water extraction as described in "An ISRU Propellant Production System to Fully Fuel a Mars
% Ascent Vehicle" (Document cited in NASA's Life support Baseline Values
% and Assumptions Document). This is assuming the extraction from regolith
% with 1.3% water content and already includes the excavator rover - Case 3
% Mass of water extraction system (kg)
Water_system_kg = 1400; 
% Power consumption of water extraction system (kW) 
Water_system_E = 45;
% Volume of water production system (m3)
Water_system_V = Water_system_kg*9.92/413.14; % Taken from Mass to volume ratio of DRA 5.0 table 6-11 3%H2O 24-hour ops
% Produced water from water extraction system (kg) 
Water_system_p = 18891+24179;   
% Cost per kg water (kg/kgWater)
Water_cost = ((Water_system_kg) + (Water_system_E.*P_eq) + (Water_system_V.*V_eq))./(Water_system_p);

% CO2 cost 
% Atmospheric acquisition subsystem from DRA5 24hr operations table 6-9
% Mass of atmospheric acquisition system (kg) %what?
CarbonDioxide_system_kg = 492.12;  
% Power consumption of atmospheric acquisition system (kW) 
CarbonDioxide_system_E = 17.863; 
% Volume of atmospheric acquisition system (m3)
CarbonDioxide_system_V = 0.66;  
% Produced Gas from atmospheric acquisition system (kg)
CarbonDioxide_system_p = 35192;
% Cost per kg CO2 (kg/kgCO2)
CarbonDioxide_cost = ((CarbonDioxide_system_kg) + (CarbonDioxide_system_E.*P_eq) + CarbonDioxide_system_V*V_eq)./(CarbonDioxide_system_p);

% N2 cost 
% Nitrogen is a wasteproduct from CO2 acquisition (together with Argon) and
% therefore carries no cost
% Cost per kg N2 (kg/kgN2)
Nitrogen_cost = 0;

% Regolith cost
% Much like Nitrogen regolith is a common wasteproduct from other systems
% (for example the current water extraction system requires 76.92 kg of
% regolith per kg of water) 
% Cost per kg regolith (kg/kg Regolith) assuming that the regolith from the
% water acquisition system is reused
Regolith_cost = 0; 

% Regolith gathering as described in "An ISRU Propellant Production System to Fully Fuel a Mars
% Ascent Vehicle" (Document cited in NASA's Life support Baseline Values
% and Assumptions Document).
% Mass of rassor excavator rover
%Excavator_rover_kg = 250;
% Power consumption of rassor excavator rover (kW) per h operation
%Excavator_rover_E = 0; 
% Produced rover regolith (kg) 
%Excavator_rover_p = 5.1028e+5; 
% Cost per kg regolith (kg/kg Regolith) 
%Regolith_cost = (Excavator_rover_kg + (Excavator_rover_E.*P_eq))./Excavator_rover_p;

% Reactor structural cost
% Mass of PBR (kg)
%PBR properties
PBR.V_liq = V_culture;   % liquid volume (m^3)
PBR.h_cylLiq = H_culture;% liquid height in cylinder (m)
PBR.r_h = 0.90;          % percent of total wall - leave room for spill over (normalised)
PBR.r_air = 0.10;        % percent of extra cylinder heigth for air (normalised)
PBR.r2d = 2;             % raise to downer intersection area ratio (normalised)
PBR.p = P_TOTAL_R_bottom * 100; % pressure inside PBR (Pa)
    
%Environment properties
environment.p_Mars = 600;    %[6 mBar] Mars atmospheric pressure (Pa)
environment.g = g;      % Martian gravity (m/s^2)
environment.rho_liq = rho_M;  % liquid density (kg/m^3)
    
% Material properties 
% PMMA
materials.PMMA.matD = 1190; % density (kg/m^3)
materials.PMMA.matR = 73; % tensile strength (MPa)
materials.PMMA.matSafety = 2; % safety coefficient
materials.PMMA.matMinS = 0.003; % min. manufacturable wall thickness (m)
% Reactor structural cost
% Mass of PBR (kg)
weightPBR = getMassPBR(materials.PMMA, environment, PBR);
Reactor_kg = weightPBR; 
% Reactor cost per m3 culture (kg)
Reactor_cost = Reactor_kg;
% Power consumption of reactor (kW) per h operation
Reactor_E = 0;

% Pumping
% Data from Vincent's documentation
% Mass of water pumps for reactor (kg) per m3 culture 
Water_pump_kg = 1.5*2;
% Power consumption of water pump (kW) (double check)
%Water_pump_E = (0.04/1000)*((((-w_out*1000)+Water_kg)*Time_harmonization)/(time*60*60))*2;
Water_pump_E = 2*25/1000;
% Mass of air pump for reactor (kg) per m3 culture 
Air_pump_kg = 1;
% Power consumption of aerating pump (kW) 
Air_pump_E = 0.5*(Oxygen_kg/(time*60*60));
% Mass of pumps (kg) per m3 of culture
Pumps_kg = Water_pump_kg + Air_pump_kg;
% Power consumption of pumps (kW) per hour of operation
Pumps_E = Air_pump_E + Water_pump_E;

% Sensors and electronics 
% Data from Vincent's documentation
% Mass of sensors and controllers (kg) per m3 of culture 
Sensors_controllers_kg = 3.29;
% Power consumption of sensors (kW)
Sensors_controllers_E = (20.49/1000);
% Cost of sensors and electronics (kg/PBR)
Sensors_electronics_cost = Sensors_controllers_kg + Sensors_controllers_E*P_eq;

% Illumination
% Mass of illumination system (kg) 
Illumination_system_kg = 18.70 + (((2*pi*Base_r*H_culture)+2*(2*pi*r_i*H_culture*0.9)))*0.945; % Calculated with lamps rather than LED so a bit overestimated
% LED light intensity to Joule (mol J-1)
LED_E = 1.66*10^-6; 
% Estimated LED utilization ratio over sunlight 
Ratio_LED_Sunlight = (50+0.5*500)/550; % Assuming 50 days of dust storms per 550 days and accounting for night time (Assumed to be half the day)
% Power consumption of illumination system (kW)
Illumination_system_E = (((mean(Iph_0_light2)/LED_E).*((2*pi*Base_r*H_culture)+2*(2*pi*r_i*H_culture*0.9)))/1000)*Ratio_LED_Sunlight; % I added the 0.9 to account for the fact that the riser is not as high as the outer shell; For vincent: why do you include the efficiency rate in your calculation? seems to me that you already have a power to light conversion
% Cost of Lighting  (kg/PBR)
Lighting_cost = Illumination_system_kg + Illumination_system_E.*P_eq;

% Cooling
% The following thermal control calculations are based on table 4-88 of NASA's Life support Baseline Values
% and Assumptions Document and Paul Zabel's ESM paper referenced within 
% Thermal control (kWth) 
Thermal_control_E = (Reactor_E + Sensors_controllers_E + Illumination_system_E + Pumps_E);
% Cost of thermal control (kg)
Thermal_control_cost = Thermal_control_E*C_eq;

% Overall ESM cost
% Mass of all photobioreactor components (kg)
Mass = Reactor_kg + Sensors_controllers_kg + Illumination_system_kg + Pumps_kg; 
% Power cost of of photobioreactor components (kg) 
Power_cost = (Reactor_E + Sensors_controllers_E + Illumination_system_E + Pumps_E)*P_eq;
% ESM cost for the photobioreactor implementation and operation
%(WITHOUT CONSUMABLES) for 10 years
ESM_PBR = Mass + Volume_cost + Power_cost + Thermal_control_cost + Crew_time_cost;
% Consumables cost (regolith, gases) for the operation time input
ESM_consumables = Regolith_kg*Regolith_cost + CarbonDioxide_kg*CarbonDioxide_cost + Nitrogen_kg*Nitrogen_cost;
% Initial water input into the photobioreactor (kg)
Water_kg  = V_culture*1000;
% ESM for Biomass (ESM with consumables in kgInput/kgBiomass)
ESM_Biomass = double((ESM_PBR + Water_kg*Water_cost + ESM_consumables*Time_harmonization)/(Biomass_kg*Time_harmonization))
% ESM for Oxygen (ESM with consumables in kgInput/kgOxygen)
ESM_Oxygen = double((ESM_PBR + Water_kg*Water_cost + ESM_consumables*Time_harmonization)/(Oxygen_kg*Time_harmonization))
% Complete ESM (ESM with consumables in kgInput/kgBiomass)
ESM_specific = double((ESM_PBR + Water_kg*Water_cost + ESM_consumables*Time_harmonization)/((Biomass_kg+Oxygen_kg)*Time_harmonization))
% Water was excluded from the time calculation because it can be reused at
% a very high efficiency. However we are missing the cost for water
% recovery (but that is DSP) - This reference should help https://ntrs.nasa.gov/citations/20160001262

%%
function massPBR = getMassPBR(material, environment, PBR)
    
    matD = material.matD;
    matR = material.matR;
    matSafety = material.matSafety;
    matMinS = material.matMinS;

    p_Mars = environment.p_Mars;
    g = environment.g;
    rho_liq = environment.rho_liq;

    V_liq = PBR.V_liq;
    h_cylLiq = PBR.h_cylLiq;
    r_h = PBR.r_h;
    r_air = PBR.r_air;
    r2d = PBR.r2d;
    
    p = PBR.p - p_Mars; % calculate effective acting pressure
    
    d_cyl = getDiameterCylinder_byVolumeHeight(V_liq, h_cylLiq);
    
    maxCylPressure = p + getHydrostaticPressure(g, rho_liq, h_cylLiq);
    minCylThickness = getCylWallThickness_ByPressure(maxCylPressure, d_cyl, matR, matSafety);
    if (minCylThickness > matMinS)
        cylThickness = minCylThickness;
    else
        cylThickness = matMinS;
    end
    totalCylHeight = h_cylLiq * (1+r_air);
    cylWallVolume = getCylWall_volume(d_cyl, totalCylHeight, cylThickness);

    h_cyl_inner = h_cylLiq * r_h;
    d_cyl_inner = getDiameterInnerCylinder(d_cyl, matMinS, r2d);
    innerCylWallVolume = getCylWall_volume(d_cyl_inner, h_cyl_inner, matMinS);
    % The inner cylinder volume is just used to calculate the weight of the
    % structure. 
    % The displaced liquid which results in the need for a bigger outer
    % cylinder diameter is not handeled in this estimation.

    topPlateThickness = getPlateWallThickness_ByPressureDiameter(...
        p, d_cyl, cylThickness, material.matR, matSafety);
    topPlateVolume = getCylinder_volume(d_cyl, topPlateThickness, cylThickness);

    basePlateThicknessSupported = matMinS;
    basePlateVolumeSupported = getCylinder_volume(d_cyl, basePlateThicknessSupported, cylThickness);

    %For calculation the baseplate with support is used 
    % since a base of regolith is assumed
    basePlateVolume = basePlateVolumeSupported;
    
    wallVolume = topPlateVolume + cylWallVolume + innerCylWallVolume + basePlateVolume;
    %wallVolume = topPlateVolume + cylWallVolume + basePlateVolume;
    massPBR = wallVolume * matD;

    fprintf('d_cyl = %.4f\n', d_cyl);
    fprintf('h_cylLiq = %.4f\n', h_cylLiq);
    fprintf('d_cyl_inner = %.4f\n', d_cyl_inner);
    fprintf('h_cyl_inner = %.4f\n', h_cyl_inner);
    fprintf('cylWallVolume = %.4f\n', cylWallVolume);
    fprintf('innerCylWallVolume = %.4f\n', innerCylWallVolume);
    fprintf('massPBR = %.4f\n', massPBR);
    fprintf('\n');
end

function d_cyl = getDiameterCylinder_byVolumeHeight(...   % (m)
        V,...    % volume (m^3)
        h)       % cylinder height (m)
    r = sqrt(V / (pi * h));
    d_cyl = 2*r;
end

function d_innerCyl = getDiameterInnerCylinder(...  (m)
        d,...           % diameter of outer cylinder (m)
        matMinS, ...    % minimal material thickness (m)
        r2d)            % raiser to downer ratio (#)
    d_innerCyl = (-4*matMinS + sqrt( (4*matMinS)^2 - 4 * (r2d + 1) * (4 * matMinS^2 - d^2) ) ) / ( 2 * (r2d + 1) );
end

function cylWallThickness = getCylWallThickness_ByPressure(...  (m)
        pressure,...    % pressure (Pa)
        d_i,... % diameter to inner wall (m)
        maxStress,...   %tensile strength (MPa)
        weaknessCoefficient)% safety coefficient
    maxStress = maxStress * 10^6; %MPa to Pa
    cylWallThickness = (pressure * d_i) / ((2 * maxStress) / weaknessCoefficient - pressure);
end

function plateWallThickness = getPlateWallThickness_ByPressureDiameter(...  (m)
        pressure,...    % pressure (Pa)
        d_i,... % diameter to inner wall (m)
        cylWallThickness,... %cylinder wall thickness (m)
        maxStress,...   %tensile strength (MPa)
        weaknessCoefficient)% safety coefficient
    maxStress = maxStress * 10^6; %MPa to Pa
    
    B = 1 - ( (3*maxStress) / (weaknessCoefficient * pressure) ) * (cylWallThickness/d_i)^2 ...
        + 3/16 * (cylWallThickness/d_i)^4 * (pressure / (maxStress / weaknessCoefficient) ) ...
        - 3/4 * cylWallThickness^2 * (2*d_i+cylWallThickness) / (d_i + cylWallThickness)^3 ;
    A = B * (1 - B * cylWallThickness / (2*(d_i + cylWallThickness)) );
    C = max( 0.40825 * A * ( (d_i + cylWallThickness) / d_i ), 0.299 * (1+ 1.7*cylWallThickness/d_i) );
    
    plateWallThickness = C * (d_i - 2 * cylWallThickness) * sqrt(pressure / (maxStress/weaknessCoefficient) );
end
%TODO calc needed material volume vs increased liquid volume

function cylWall_volume = getCylWall_volume(... (m^3)
        d_i,... % diameter to inner wall (m)
        h,...   % cylinder heigth (m)
        s)      % wall thickness (m)
    cylWall_volume = pi * h * ( (d_i/2+s)^2 - (d_i/2)^2 );
end

function cylinder_volume = getCylinder_volume(... (m^3)
        d_i,... % diameter of tank (m)
        h,...   % cylinder heigth (m)
        s)      % wall thickness (m)
    cylinder_volume = pi * h * (d_i/2+s)^2;
end

function hydrostaticPressure = getHydrostaticPressure(...   (Pa)
        gravity,...         % m/s^2
        liquidDensity,...   % kg/m^3
        height)             % m
    hydrostaticPressure = gravity * liquidDensity * height;
end