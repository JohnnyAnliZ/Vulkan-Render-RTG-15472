#ifndef SHADER_UTIL
#define SHADER_UTIL

float cubic_spline_utility(
    float x,//interpolation point
    float x0, float y0, float m0,
    float x1, float y1, float m1
){//convert from f(x)
    float dx = x1 - x0;
    float t = (x - x0)/dx;
    t = clamp(t, 0.0, 1.0);

    float t_sqr = t * t;
    float t_cube = t_sqr * t;

    //hermite basis are derived from the inverse of the 4 x 4 matrix that transforms a,b,c,d into f(x=x0), f(x=x1), f'(x=x0), f'(x=x1)
    //which are, in the context of H(t), H(t=0) = y0,  H(t=1) = y1, H'(t=0)m0 * dx, H'(t=1) =m1 *dx
    //(dx here is actually the same as dx/dt, chain rule for going from f(x) to H(t))

    //hermite basis are:
    float h0 =  2 * t_cube - 3 * t_sqr + 1;
    float h1 =  -2 * t_cube + 3 * t_sqr;
    float h2 =  t_cube - 2 * t_sqr + t;
    float h3 =  t_cube -  t_sqr;

    return h0 * y0 + h1 *y1 + h2 * m0 * dx + h3 * m1 * dx;

}

#endif
