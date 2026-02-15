#include "mat4.hpp"


//haven't tested this
mat4 mat4::inverse()
{
    //
    // Inversion by Cramer's rule.  Code taken from an Intel publication
    //
    float Result[4][4];
    float tmp[12]; /* temp array for pairs */
    float src[16]; /* array of transpose source matrix */
    float det; /* determinant */
    /* transpose matrix */
    for (uint32_t i = 0; i < 4; i++)
    {
        src[i + 0 ] = data.at(i * 4 + 0);
        src[i + 4 ] = data.at(i * 4 + 1);
        src[i + 8 ] = data.at(i * 4 + 2);
        src[i + 12] = data.at(i * 4 + 3);
    }
    /* calculate pairs for first 8 elements (cofactors) */
    tmp[0] = src[10] * src[15];
    tmp[1] = src[11] * src[14];
    tmp[2] = src[9] * src[15];
    tmp[3] = src[11] * src[13];
    tmp[4] = src[9] * src[14];
    tmp[5] = src[10] * src[13];
    tmp[6] = src[8] * src[15];
    tmp[7] = src[11] * src[12];
    tmp[8] = src[8] * src[14];
    tmp[9] = src[10] * src[12];
    tmp[10] = src[8] * src[13];
    tmp[11] = src[9] * src[12];
    /* calculate first 8 elements (cofactors) */
    Result[0][0] = tmp[0]*src[5] + tmp[3]*src[6] + tmp[4]*src[7];
    Result[0][0] -= tmp[1]*src[5] + tmp[2]*src[6] + tmp[5]*src[7];
    Result[0][1] = tmp[1]*src[4] + tmp[6]*src[6] + tmp[9]*src[7];
    Result[0][1] -= tmp[0]*src[4] + tmp[7]*src[6] + tmp[8]*src[7];
    Result[0][2] = tmp[2]*src[4] + tmp[7]*src[5] + tmp[10]*src[7];
    Result[0][2] -= tmp[3]*src[4] + tmp[6]*src[5] + tmp[11]*src[7];
    Result[0][3] = tmp[5]*src[4] + tmp[8]*src[5] + tmp[11]*src[6];
    Result[0][3] -= tmp[4]*src[4] + tmp[9]*src[5] + tmp[10]*src[6];
    Result[1][0] = tmp[1]*src[1] + tmp[2]*src[2] + tmp[5]*src[3];
    Result[1][0] -= tmp[0]*src[1] + tmp[3]*src[2] + tmp[4]*src[3];
    Result[1][1] = tmp[0]*src[0] + tmp[7]*src[2] + tmp[8]*src[3];
    Result[1][1] -= tmp[1]*src[0] + tmp[6]*src[2] + tmp[9]*src[3];
    Result[1][2] = tmp[3]*src[0] + tmp[6]*src[1] + tmp[11]*src[3];
    Result[1][2] -= tmp[2]*src[0] + tmp[7]*src[1] + tmp[10]*src[3];
    Result[1][3] = tmp[4]*src[0] + tmp[9]*src[1] + tmp[10]*src[2];
    Result[1][3] -= tmp[5]*src[0] + tmp[8]*src[1] + tmp[11]*src[2];
    /* calculate pairs for second 8 elements (cofactors) */
    tmp[0] = src[2]*src[7];
    tmp[1] = src[3]*src[6];
    tmp[2] = src[1]*src[7];
    tmp[3] = src[3]*src[5];
    tmp[4] = src[1]*src[6];
    tmp[5] = src[2]*src[5];

    tmp[6] = src[0]*src[7];
    tmp[7] = src[3]*src[4];
    tmp[8] = src[0]*src[6];
    tmp[9] = src[2]*src[4];
    tmp[10] = src[0]*src[5];
    tmp[11] = src[1]*src[4];
    /* calculate second 8 elements (cofactors) */
    Result[2][0] = tmp[0]*src[13] + tmp[3]*src[14] + tmp[4]*src[15];
    Result[2][0] -= tmp[1]*src[13] + tmp[2]*src[14] + tmp[5]*src[15];
    Result[2][1] = tmp[1]*src[12] + tmp[6]*src[14] + tmp[9]*src[15];
    Result[2][1] -= tmp[0]*src[12] + tmp[7]*src[14] + tmp[8]*src[15];
    Result[2][2] = tmp[2]*src[12] + tmp[7]*src[13] + tmp[10]*src[15];
    Result[2][2] -= tmp[3]*src[12] + tmp[6]*src[13] + tmp[11]*src[15];
    Result[2][3] = tmp[5]*src[12] + tmp[8]*src[13] + tmp[11]*src[14];
    Result[2][3] -= tmp[4]*src[12] + tmp[9]*src[13] + tmp[10]*src[14];
    Result[3][0] = tmp[2]*src[10] + tmp[5]*src[11] + tmp[1]*src[9];
    Result[3][0] -= tmp[4]*src[11] + tmp[0]*src[9] + tmp[3]*src[10];
    Result[3][1] = tmp[8]*src[11] + tmp[0]*src[8] + tmp[7]*src[10];
    Result[3][1] -= tmp[6]*src[10] + tmp[9]*src[11] + tmp[1]*src[8];
    Result[3][2] = tmp[6]*src[9] + tmp[11]*src[11] + tmp[3]*src[8];
    Result[3][2] -= tmp[10]*src[11] + tmp[2]*src[8] + tmp[7]*src[9];
    Result[3][3] = tmp[10]*src[10] + tmp[4]*src[8] + tmp[9]*src[9];
    Result[3][3] -= tmp[8]*src[9] + tmp[11]*src[10] + tmp[5]*src[8];
    /* calculate determinant */
    det=src[0]*Result[0][0]+src[1]*Result[0][1]+src[2]*Result[0][2]+src[3]*Result[0][3];
    /* calculate matrix inverse */
    det = 1.0f / det;

    mat4 FloatResult;
    for (uint32_t i = 0; i < 4; i++)
    {
        for (uint32_t j = 0; j < 4; j++)
        {
            FloatResult[i*4+j] = float(Result[j][i] * det);//transposing again, writing back to column-major format
        }
    }
    return FloatResult;

    //
    // Inversion by LU decomposition, alternate implementation
    //
    /*int i, j, k;

    for (i = 1; i < 4; i++)
    {
        _Entries[0][i] /= _Entries[0][0];
    }

    for (i = 1; i < 4; i++)
    {
        for (j = i; j < 4; j++)
        {
            float sum = 0.0;
            for (k = 0; k < i; k++)
            {
                sum += _Entries[j][k] * _Entries[k][i];
            }
            _Entries[j][i] -= sum;
        }
        if (i == 4-1) continue;
        for (j=i+1; j < 4; j++)
        {
            float sum = 0.0;
            for (int k = 0; k < i; k++)
                sum += _Entries[i][k]*_Entries[k][j];
            _Entries[i][j] = 
               (_Entries[i][j]-sum) / _Entries[i][i];
        }
    }

    //
    // Invert L
    //
    for ( i = 0; i < 4; i++ )
    {
        for ( int j = i; j < 4; j++ )
        {
            float x = 1.0;
            if ( i != j )
            {
                x = 0.0;
                for ( int k = i; k < j; k++ ) 
                    x -= _Entries[j][k]*_Entries[k][i];
            }
            _Entries[j][i] = x / _Entries[j][j];
        }
    }

    //
    // Invert U
    //
    for ( i = 0; i < 4; i++ )
    {
        for ( j = i; j < 4; j++ )
        {
            if ( i == j ) continue;
            float sum = 0.0;
            for ( int k = i; k < j; k++ )
                sum += _Entries[k][j]*( (i==k) ? 1.0f : _Entries[i][k] );
            _Entries[i][j] = -sum;
        }
    }

    //
    // Final Inversion
    //
    for ( i = 0; i < 4; i++ )
    {
        for ( int j = 0; j < 4; j++ )
        {
            float sum = 0.0;
            for ( int k = ((i>j)?i:j); k < 4; k++ )  
                sum += ((j==k)?1.0f:_Entries[j][k])*_Entries[k][i];
            _Entries[j][i] = sum;
        }
    }*/
}


//taken and modified from http://number-none.com/product/Understanding%20Slerp,%20Then%20Not%20Using%20It/
quat quat::slerp(quat const &v0, quat const &v1, float t){
    // v0 and v1 should be unit length or else
    // something broken will happen.

    // Compute the cosine of the angle between the two vectors.
    float dot_product = dot(v0, v1);

    const float DOT_THRESHOLD = 0.9995f;
    if (dot_product > DOT_THRESHOLD) {
        // If the inputs are too close for comfort, linearly interpolate
        // and normalize the result.

        quat result = v0 + t*(v1 - v0);
        result.normalize();
        return result;
    }

    dot_product = std::min(dot_product, 1.0f);
    dot_product = std::max(dot_product, -1.0f);// Robustness: Stay within domain of acos()
    
    float theta_0 = acos(dot_product);  // theta_0 = angle between input vectors
    float theta = theta_0*t;    // theta = angle between v0 and result 

    quat v2 = v1 - v0*dot_product;
    v2.normalize();              // { v0, v2 } is now an orthonormal basis

    return v0*cos(theta) + v2*sin(theta);
}