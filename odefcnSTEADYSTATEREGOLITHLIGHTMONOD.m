function dcdt = odefcnSTEADYSTATEREGOLITHLIGHTMONOD(t,c,R_P,BET,C_R,mu_max,K_P,P_COEFF,f_ClO4,V_culture,Cx_set,f_T,mu_L_avg_SS,Regolith_shading,Ki_P,C_P0)
    % This checkpoint was added as the ODE45 was occasionally calculating
    % negative phosphorus concentrations which is physically impossible
    % Check if c is negative
    if c < 0
        % If c is negative, set it to zero
        c = 0;
        % Also set f_w_out to zero to ensure that dcdt is zero
        f_w_out = 0;
    else
        % If c is non-negative, calculate f_w_out as normal (m3 s-1)
        f_w_out = -V_culture*min(((mu_max.*c)./(K_P+(c.*(1+c./Ki_P)))),mu_L_avg_SS)*f_ClO4*f_T;
    end
    
    % Phosphorus concentration over time (dCPdt)
    % Calculate dcdt as normal, using the potentially modified value of c
    dcdt = (-f_w_out*C_P0)./V_culture + (f_w_out*c)./V_culture + R_P.*BET.*C_R + Cx_set.*min(((mu_max.*c)./(K_P+(c.*(1+c./Ki_P)))),mu_L_avg_SS)*f_ClO4*f_T*(-P_COEFF);
end
