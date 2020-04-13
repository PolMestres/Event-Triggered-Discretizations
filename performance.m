%% Performance-Based Algorithm
%% Self-Triggered

function [x1valuesst,x2valuesst,normxst,lyapvaluesst,fvaluesst] = performance(f,gradf,optim,mu,L,a,lyapunov,Xhba,s,alfa,niter,tol,x0,v0)
    disp('performance')
    xhat = x0;
    vhat = v0;
    phat = [xhat,vhat];
    i = 1;
    while ((norm(gradf(xhat)) > tol) && (i <= niter))
        x1valuesst(i) = xhat(1);
        x2valuesst(i) = xhat(2);
        normxst(i) = norm(xhat(1),xhat(2));
        lyapvaluesst(i) = lyapunov(xhat,vhat);
        fvaluesst(i) = f(xhat);
        
        t1 = (1+sqrt(mu*s))*L*norm(vhat)^2;
        t2 = 2*sqrt(mu)*(1+sqrt(mu*s))*dot(gradf(xhat+a*vhat),vhat);
        t3 = 2*mu*norm(vhat)^2;
        t4 = (1+sqrt(mu*s))^2 * norm(gradf(xhat+a*vhat))^2;
        Ast = t1+t2+t3+t4;
        
        u1 = -alfa*sqrt(mu)*norm(vhat)^2;
        u2 = -alfa*sqrt(mu)*(1+sqrt(mu*s))/L * norm(gradf(xhat+a*vhat))^2;
        u3 = alfa*sqrt(mu)*(1+sqrt(mu*s))*a*dot(vhat,gradf(xhat+a*vhat));
        u4 = alfa*(1+sqrt(mu*s))*dot(vhat,gradf(xhat)-gradf(xhat+a*vhat));
        Bst = u1+u2+u3+u4;
        
        v1 = alfa*(1+sqrt(mu*s))*L/2 *norm(vhat)^2;
        v2 = alfa/4 *norm(-2*sqrt(mu)*vhat-(1+sqrt(mu*s))*gradf(xhat+a*vhat))^2;
        v3 = alfa/4 * (1+sqrt(mu*s))^2 * norm(gradf(xhat+a*vhat))^2;
        Cst = v1+v2+v3;
        
        w1 = (1+sqrt(mu*s))*dot(gradf(xhat)-gradf(xhat+a*vhat),vhat);
        w2 = (3*alfa/4 - sqrt(mu)-mu*sqrt(mu)*(1+sqrt(mu*s))*a^2 /2)*norm(vhat)^2;
        w3 = -(mu/2 * sqrt(mu)*(1+sqrt(mu*s))-2*alfa*mu)*(1/L^2)*norm(gradf(xhat))^2;
        w4 = sqrt(mu)*(1+sqrt(mu*s))*a*norm(gradf(xhat))*norm(vhat);
        w5 = a*sqrt(mu)*(1+sqrt(mu*s))*dot(vhat,gradf(xhat+a*vhat));
        w6 = -(1+sqrt(mu*s))*(sqrt(mu)-alfa)/(2*L) * norm(gradf(xhat))^2;
        w7 = sqrt(mu)*(1+sqrt(mu*s))*(f(xhat)-f(xhat+a*vhat));
        Dst = w1+w2+w3+w4+w5+w6+w7;
        
        k1 = Cst;
        k2 = Ast+Bst;
        k3 = Dst;
        
        %perfst = @(t)(exp(alfa*t)*(k1/alfa^2 *t^2 + (k2/alfa - 2*k1/alfa^2)*t + k3/alfa - k2/alfa^2 + 2*k1/alfa^3)-(k3/alfa - k2/alfa^2 + 2*k1/alfa^3));
        perfst = @(t)integral(@(tau)(exp(alfa*tau).*(k1*tau.^2 + k2*tau + k3)),0,t);
        
        %to find the interval where the solution is we use the fact that
        %stepst(p) is less than stepstp(p)
        
        cnt = 2;
        stepst = (-(Ast+Bst)+sqrt((Ast+Bst)^2 - 4*Cst*Dst))/(2*Cst);
        
        while(perfst(cnt*stepst) < 0)
            cnt = cnt + 1;
        end
        
        tk = fzero(perfst,[stepst,cnt*stepst]);
        
        phat = phat + tk*Xhba(xhat,vhat,a);
        xhat = [phat(1),phat(2)];
        vhat = [phat(3),phat(4)];
        i = i+1;
    end
end
