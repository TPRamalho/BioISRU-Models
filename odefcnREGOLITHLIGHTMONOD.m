function dydt = odefcnREGOLITHLIGHTMONOD(t,y,R_P,BET,C_R,mu_max,K_P,P_COEFF,f_ClO4,f_T,a_r,a_x,Iph_0,r_adjust,K_L,Base_r,r_i,LG1_r,LG2_r,Regolith_shading,Ki_P)
    dydt = zeros(2,1);
    if Regolith_shading == 1
        Atten_reg = a_r.*C_R;
    else
        Atten_reg = 0;
    end
    if y(1) < 0
        % If c is negative, set it to zero
        y(1) = 0;
    end
    Atten_x = a_x.*y(2);
    
    if exist('LG3_r','var') == 0
        % Light intensity calculations (draft tube and outer shell)
        % All light intensities are in molph m-2 s-1
        r_center = r_adjust; % Adjustments to LG1 and LG2
        r1_RI = linspace(LG1_r,r_center,1000); % riser path 
        r1_DC = linspace(LG1_r,LG2_r,1000); % downcomer path LG1
        r2_DC = linspace(LG2_r,LG1_r,1000); % dowcnomer path LG2   
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
    % Monod calculation of growth rate as a function of ligh intensity 
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
    mu_L_avg = (r_i/Base_r).*mu_L_RI_avg + ((Base_r-r_i)/Base_r).*mu_L_DC_avg;
    % Growth rate as a function of phosphorus
    mu_P = (mu_max.*y(1))./(K_P+(y(1).*(1+y(1)./Ki_P)));
    % Biomass concentration over time (dCxdt - molx m-3 s-1)
    dydt(2) = min(mu_L_avg,mu_P).*y(2).*f_ClO4.*f_T;
    % Phosphorus concentration over time (dCPdt - molP m-3 s-1)
    dydt(1) = R_P.*BET.*C_R + dydt(2).*(-P_COEFF);
end