function [x1values, x2values, normx, lyapvalues, fvalues] = fohdefinitiu(f, gradf, optim, mu, L, s, alfa, niter, tol, x0, v0, lyapunov)
    disp('foh-definitiu')
    xhat = x0;
    vhat = v0;
    i = 1;
    while((norm(gradf(xhat)) > tol) && (i <= niter))
        x1values(i) = xhat(1);
        x2values(i) = xhat(2);
        normx(i) = norm(xhat(1),xhat(2));
        lyapvalues(i) = lyapunov(xhat,vhat);
        fvalues(i) = f(xhat);
        
        K1 = -vhat/(2*sqrt(mu))-(1+sqrt(mu*s))/(4*mu)*gradf(xhat);
        K2 = xhat + vhat/(2*sqrt(mu))+(1+sqrt(mu*s))/(4*mu)*gradf(xhat);
        
        xt = @(t)(K1*exp(-2*sqrt(mu)*t)-(1+sqrt(mu*s))/(2*sqrt(mu))*gradf(xhat)*t + K2);
        vt = @(t)(-2*sqrt(mu)*K1*exp(-2*sqrt(mu)*t)-(1+sqrt(mu*s))/(2*sqrt(mu))*gradf(xhat));
        
        q1 = @(t)((1+sqrt(mu*s))*dot(gradf(xt(t))-gradf(xt(0)),vt(t)));
        q2 = @(t)(sqrt(mu)*dot(xt(t)-xt(0),vt(t)));
        q3 = @(t)(-2*sqrt(mu)*dot(vt(t)-vt(0),vt(t)));
        q4 = @(t)(-(1+sqrt(mu*s))*dot(vt(t)-vt(0),gradf(xt(0))));
        
        termh1 = @(t)(q1(t)+q2(t)+q3(t)+q4(t));
        %termh1(0)
        
        w1 = @(t)((1+sqrt(mu*s))*dot(gradf(xhat)+sqrt(mu)*vhat,vt(t)-vhat));
        w2 = @(t)(dot(vhat,2*sqrt(mu)*(vt(t)-vhat)));
        
        termh2 = @(t)(w1(t)+w2(t));
        %termh2(0)
        
        r1 = @(t)((1+sqrt(mu*s))*(f(xt(t))-f(xhat)));
        r2 = @(t)(1/2 * (norm(vt(t))^2 - norm(vhat)^2));
        r3 = @(t)(1/2 * norm(vt(t)-vhat + 2*sqrt(mu)*(xt(t)-xhat))^2);
        r4 = @(t)(2*dot(vt(t)-vhat+4*sqrt(mu)*(xt(t)-xhat),vhat));
        r5 = @(t)(2*t*(1+sqrt(mu*s))/sqrt(mu) * norm(gradf(xhat))^2);
        
        termh3 = @(t)(r1(t)+r2(t)+r3(t)+r4(t)+r5(t));
        %termh3(0)
        
        termh4 = (alfa*3/4 - sqrt(mu))*norm(vhat)^2 + ((1+sqrt(mu*s))*(alfa-sqrt(mu))/(2*L) + (alfa*2*mu-sqrt(mu)*(1+sqrt(mu*s))*mu/2)*1/L^2)*norm(gradf(xhat))^2;
        %termh4
        
        gET = @(t)(termh1(t)+termh2(t)+termh3(t)+termh4);
        
        right = 0.5;
        while(gET(right) < 0)
            right = right*1.5;
        end
        
        %gET(0)
        %gET(right)
        
        tk = fzero(gET,[0,right]);
        
        xhat = xt(tk);
        vhat = vt(tk);
        i = i+1;
    end  
end