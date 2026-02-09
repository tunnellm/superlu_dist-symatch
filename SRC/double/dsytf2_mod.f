*> \brief \b DSYTF2 computes the factorization of a real symmetric indefinite matrix, using the diagonal pivoting method (unblocked algorithm).
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at
*            http://www.netlib.org/lapack/explore-html/
*
*> Download DSYTF2 + dependencies
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dsytf2.f">
*> [TGZ]</a>
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dsytf2.f">
*> [ZIP]</a>
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dsytf2.f">
*> [TXT]</a>
*
*  Definition:
*  ===========
*
*       SUBROUTINE DSYTF2( UPLO, N, A, LDA, IPIV, INFO )
*
*       .. Scalar Arguments ..
*       CHARACTER          UPLO
*       INTEGER            INFO, LDA, N
*       ..
*       .. Array Arguments ..
*       INTEGER            IPIV( * )
*       DOUBLE PRECISION   A( LDA, * )
*       ..
*
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*> DSYTF2 computes the factorization of a real symmetric matrix A using
*> the Bunch-Kaufman diagonal pivoting method:
*>
*>    A = U*D*U**T  or  A = L*D*L**T
*>
*> where U (or L) is a product of permutation and unit upper (lower)
*> triangular matrices, U**T is the transpose of U, and D is symmetric and
*> block diagonal with 1-by-1 and 2-by-2 diagonal blocks.
*>
*> This is the unblocked version of the algorithm, calling Level 2 BLAS.
*> \endverbatim
*
*  Arguments:
*  ==========
*
*> \param[in] UPLO
*> \verbatim
*>          UPLO is CHARACTER*1
*>          Specifies whether the upper or lower triangular part of the
*>          symmetric matrix A is stored:
*>          = 'U':  Upper triangular
*>          = 'L':  Lower triangular
*> \endverbatim
*>
*> \param[in] N
*> \verbatim
*>          N is INTEGER
*>          The order of the matrix A.  N >= 0.
*> \endverbatim
*>
*> \param[in,out] A
*> \verbatim
*>          A is DOUBLE PRECISION array, dimension (LDA,N)
*>          On entry, the symmetric matrix A.  If UPLO = 'U', the leading
*>          n-by-n upper triangular part of A contains the upper
*>          triangular part of the matrix A, and the strictly lower
*>          triangular part of A is not referenced.  If UPLO = 'L', the
*>          leading n-by-n lower triangular part of A contains the lower
*>          triangular part of the matrix A, and the strictly upper
*>          triangular part of A is not referenced.
*>
*>          On exit, the block diagonal matrix D and the multipliers used
*>          to obtain the factor U or L (see below for further details).
*> \endverbatim
*>
*> \param[in] LDA
*> \verbatim
*>          LDA is INTEGER
*>          The leading dimension of the array A.  LDA >= max(1,N).
*> \endverbatim
*>
*> \param[out] IPIV
*> \verbatim
*>          IPIV is INTEGER array, dimension (N)
*>          Details of the interchanges and the block structure of D.
*>
*>          If UPLO = 'U':
*>             If IPIV(k) > 0, then rows and columns k and IPIV(k) were
*>             interchanged and D(k,k) is a 1-by-1 diagonal block.
*>
*>             If IPIV(k) = IPIV(k-1) < 0, then rows and columns
*>             k-1 and -IPIV(k) were interchanged and D(k-1:k,k-1:k)
*>             is a 2-by-2 diagonal block.
*>
*>          If UPLO = 'L':
*>             If IPIV(k) > 0, then rows and columns k and IPIV(k) were
*>             interchanged and D(k,k) is a 1-by-1 diagonal block.
*>
*>             If IPIV(k) = IPIV(k+1) < 0, then rows and columns
*>             k+1 and -IPIV(k) were interchanged and D(k:k+1,k:k+1)
*>             is a 2-by-2 diagonal block.
*> \endverbatim
*>
*> \param[out] INFO
*> \verbatim
*>          INFO is INTEGER
*>          = 0: successful exit
*>          < 0: if INFO = -k, the k-th argument had an illegal value
*>          > 0: if INFO = k, D(k,k) is exactly zero.  The factorization
*>               has been completed, but the block diagonal matrix D is
*>               exactly singular, and division by zero will occur if it
*>               is used to solve a system of equations.
*> \endverbatim
*
*  Authors:
*  ========
*
*> \author Univ. of Tennessee
*> \author Univ. of California Berkeley
*> \author Univ. of Colorado Denver
*> \author NAG Ltd.
*
*> \ingroup hetf2
*
*> \par Further Details:
*  =====================
*>
*> \verbatim
*>
*>  If UPLO = 'U', then A = U*D*U**T, where
*>     U = P(n)*U(n)* ... *P(k)U(k)* ...,
*>  i.e., U is a product of terms P(k)*U(k), where k decreases from n to
*>  1 in steps of 1 or 2, and D is a block diagonal matrix with 1-by-1
*>  and 2-by-2 diagonal blocks D(k).  P(k) is a permutation matrix as
*>  defined by IPIV(k), and U(k) is a unit upper triangular matrix, such
*>  that if the diagonal block D(k) is of order s (s = 1 or 2), then
*>
*>             (   I    v    0   )   k-s
*>     U(k) =  (   0    I    0   )   s
*>             (   0    0    I   )   n-k
*>                k-s   s   n-k
*>
*>  If s = 1, D(k) overwrites A(k,k), and v overwrites A(1:k-1,k).
*>  If s = 2, the upper triangle of D(k) overwrites A(k-1,k-1), A(k-1,k),
*>  and A(k,k), and v overwrites A(1:k-2,k-1:k).
*>
*>  If UPLO = 'L', then A = L*D*L**T, where
*>     L = P(1)*L(1)* ... *P(k)*L(k)* ...,
*>  i.e., L is a product of terms P(k)*L(k), where k increases from 1 to
*>  n in steps of 1 or 2, and D is a block diagonal matrix with 1-by-1
*>  and 2-by-2 diagonal blocks D(k).  P(k) is a permutation matrix as
*>  defined by IPIV(k), and L(k) is a unit lower triangular matrix, such
*>  that if the diagonal block D(k) is of order s (s = 1 or 2), then
*>
*>             (   I    0     0   )  k-1
*>     L(k) =  (   0    I     0   )  s
*>             (   0    v     I   )  n-k-s+1
*>                k-1   s  n-k-s+1
*>
*>  If s = 1, D(k) overwrites A(k,k), and v overwrites A(k+1:n,k).
*>  If s = 2, the lower triangle of D(k) overwrites A(k,k), A(k+1,k),
*>  and A(k+1,k+1), and v overwrites A(k+2:n,k:k+1).
*> \endverbatim
*
*> \par Contributors:
*  ==================
*>
*> \verbatim
*>
*>  09-29-06 - patch from
*>    Bobby Cheng, MathWorks
*>
*>    Replace l.204 and l.372
*>         IF( MAX( ABSAKK, COLMAX ).EQ.ZERO ) THEN
*>    by
*>         IF( (MAX( ABSAKK, COLMAX ).EQ.ZERO) .OR. DISNAN(ABSAKK) ) THEN
*>
*>  01-01-96 - Based on modifications by
*>    J. Lewis, Boeing Computer Services Company
*>    A. Petitet, Computer Science Dept., Univ. of Tenn., Knoxville, USA
*>  1-96 - Based on modifications by J. Lewis, Boeing Computer Services
*>         Company
*> \endverbatim
*
*  =====================================================================
      SUBROUTINE dsytf2_mod( UPLO, N, A, LDA, THRESH, IPIV, INFO,
     $ NTINY, N2x2)
*
*  -- LAPACK computational routine --
*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*
*     .. Scalar Arguments ..
      CHARACTER          UPLO
      INTEGER            INFO, LDA, N, NTINY, N2x2
*     ..
*     .. Array Arguments ..
      INTEGER            IPIV( * )
      DOUBLE PRECISION   A( LDA, * )
      DOUBLE PRECISION   THRESH
*     ..
*
*  =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ZERO, ONE
      parameter( zero = 0.0d+0, one = 1.0d+0 )
      DOUBLE PRECISION   EIGHT, SEVTEN
      parameter( eight = 8.0d+0, sevten = 17.0d+0 )
*     ..
*     .. Local Scalars ..
      LOGICAL            UPPER
      INTEGER            I, IMAX, J, JMAX, K, KK, KP, KSTEP
      DOUBLE PRECISION   ABSAKK, ALPHA, COLMAX, D11, D12, D21, D22, R1,
     $                   ROWMAX, T, WK, WKM1, WKP1
*     ..
*     .. External Functions ..
      LOGICAL            LSAME, DISNAN
      INTEGER            IDAMAX
      EXTERNAL           lsame, idamax, disnan
*     ..
*     .. External Subroutines ..
      EXTERNAL           dscal, dswap, dsyr, xerbla
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          abs, max, sqrt
*     ..
*     .. Executable Statements ..
*
*     Test the input parameters.
*
      info = 0
      upper = lsame( uplo, 'U' )
      IF( .NOT.upper .AND. .NOT.lsame( uplo, 'L' ) ) THEN
         info = -1
      ELSE IF( n.LT.0 ) THEN
         info = -2
      ELSE IF( lda.LT.max( 1, n ) ) THEN
         info = -4
      END IF
      IF( info.NE.0 ) THEN
         CALL xerbla( 'DSYTF2_MOD', -info )
         RETURN
      END IF
*
*     Initialize ALPHA for use in choosing pivot block size.
*
      alpha = ( one+sqrt( sevten ) ) / eight
*
      IF( upper ) THEN
*
*        Factorize A as U*D*U**T using the upper triangle of A
*
*        K is the main loop index, decreasing from N to 1 in steps of
*        1 or 2
*
         k = n
   10    CONTINUE
*
*        If K < 1, exit from loop
*
         IF( k.LT.1 )
     $      GO TO 70
         kstep = 1
*
*        Determine rows and columns to be interchanged and whether
*        a 1-by-1 or 2-by-2 pivot block will be used
*
         absakk = abs( a( k, k ) )
         

         if(absakk<thresh)then
*            PRINT *, 'absakk in dsytf2_1', absakk, thresh, ntiny
            if(a( k, k )>0)then
               a( k, k ) = thresh 
            else
               a( k, k ) = -thresh 
            endif
            absakk = abs( a( k, k ) )
            ntiny=ntiny+1
         endif

*
*        IMAX is the row-index of the largest off-diagonal element in
*        column K, and COLMAX is its absolute value.
*        Determine both COLMAX and IMAX.
*
         IF( k.GT.1 ) THEN
            imax = idamax( k-1, a( 1, k ), 1 )
            colmax = abs( a( imax, k ) )
         ELSE
            colmax = zero
         END IF

*
         IF( (max( absakk, colmax ).EQ.zero) .OR.
     $       disnan(absakk) ) THEN
*
*           Column K is zero or underflow, or contains a NaN:
*           set INFO and continue
*  
            IF( info.EQ.0 )
     $         info = k
            kp = k
         ELSE
            IF( absakk.GE.alpha*colmax ) THEN
*
*              no interchange, use 1-by-1 pivot block
*
               kp = k
            ELSE
*
*              JMAX is the column-index of the largest off-diagonal
*              element in row IMAX, and ROWMAX is its absolute value
*
               jmax = imax + idamax( k-imax, a( imax, imax+1 ), lda )
               rowmax = abs( a( imax, jmax ) )
               IF( imax.GT.1 ) THEN
                  jmax = idamax( imax-1, a( 1, imax ), 1 )
                  rowmax = max( rowmax, abs( a( jmax, imax ) ) )
               END IF
*
               IF( absakk.GE.alpha*colmax*( colmax / rowmax ) ) THEN
*
*                 no interchange, use 1-by-1 pivot block
*
                  kp = k
               ELSE IF( abs( a( imax, imax ) ).GE.alpha*rowmax ) THEN
*
*                 interchange rows and columns K and IMAX, use 1-by-1
*                 pivot block
*
                  kp = imax
               ELSE
*
*                 interchange rows and columns K-1 and IMAX, use 2-by-2
*                 pivot block
*
                  kp = imax
                  kstep = 2
                  n2x2 = n2x2 + 1
               END IF
            END IF
*
            kk = k - kstep + 1
            IF( kp.NE.kk ) THEN
*
*              Interchange rows and columns KK and KP in the leading
*              submatrix A(1:k,1:k)
*
               CALL dswap( kp-1, a( 1, kk ), 1, a( 1, kp ), 1 )
               CALL dswap( kk-kp-1, a( kp+1, kk ), 1, a( kp, kp+1 ),
     $                     lda )
               t = a( kk, kk )
               a( kk, kk ) = a( kp, kp )
               a( kp, kp ) = t
               IF( kstep.EQ.2 ) THEN
                  t = a( k-1, k )
                  a( k-1, k ) = a( kp, k )
                  a( kp, k ) = t
               END IF
            END IF
*
*           Update the leading submatrix
*
            IF( kstep.EQ.1 ) THEN
*
*              1-by-1 pivot block D(k): column k now holds
*
*              W(k) = U(k)*D(k)
*
*              where U(k) is the k-th column of U
*
*              Perform a rank-1 update of A(1:k-1,1:k-1) as
*
*              A := A - U(k)*D(k)*U(k)**T = A - W(k)*1/D(k)*W(k)**T
*
               r1 = one / a( k, k )
               CALL dsyr( uplo, k-1, -r1, a( 1, k ), 1, a, lda )
*
*              Store U(k) in column k
*
               CALL dscal( k-1, r1, a( 1, k ), 1 )
            ELSE
*
*              2-by-2 pivot block D(k): columns k and k-1 now hold
*
*              ( W(k-1) W(k) ) = ( U(k-1) U(k) )*D(k)
*
*              where U(k) and U(k-1) are the k-th and (k-1)-th columns
*              of U
*
*              Perform a rank-2 update of A(1:k-2,1:k-2) as
*
*              A := A - ( U(k-1) U(k) )*D(k)*( U(k-1) U(k) )**T
*                 = A - ( W(k-1) W(k) )*inv(D(k))*( W(k-1) W(k) )**T
*
               IF( k.GT.2 ) THEN
*
                  d12 = a( k-1, k )
                  d22 = a( k-1, k-1 ) / d12
                  d11 = a( k, k ) / d12
                  t = one / ( d11*d22-one )
                  d12 = t / d12
*
                  DO 30 j = k - 2, 1, -1
                     wkm1 = d12*( d11*a( j, k-1 )-a( j, k ) )
                     wk = d12*( d22*a( j, k )-a( j, k-1 ) )
                     DO 20 i = j, 1, -1
                        a( i, j ) = a( i, j ) - a( i, k )*wk -
     $                              a( i, k-1 )*wkm1
   20                CONTINUE
                     a( j, k ) = wk
                     a( j, k-1 ) = wkm1
   30             CONTINUE
*
               END IF
*
            END IF
         END IF
*
*        Store details of the interchanges in IPIV
*
         IF( kstep.EQ.1 ) THEN
            ipiv( k ) = kp
         ELSE
            ipiv( k ) = -kp
            ipiv( k-1 ) = -kp
         END IF
*
*        Decrease K and return to the start of the main loop
*
         k = k - kstep
         GO TO 10
*
      ELSE
*
*        Factorize A as L*D*L**T using the lower triangle of A
*
*        K is the main loop index, increasing from 1 to N in steps of
*        1 or 2
*
         k = 1
   40    CONTINUE
*
*        If K > N, exit from loop
*
         IF( k.GT.n )
     $      GO TO 70
         kstep = 1
*
*        Determine rows and columns to be interchanged and whether
*        a 1-by-1 or 2-by-2 pivot block will be used
*
         absakk = abs( a( k, k ) )

         if(absakk<thresh)then
*            PRINT *, 'absakk in dsytf2_2', absakk, thresh, ntiny
            if(a( k, k )>0)then
               a( k, k ) = thresh 
            else
               a( k, k ) = -thresh 
            endif
            absakk = abs( a( k, k ) )
            ntiny=ntiny+1
         endif


*
*        IMAX is the row-index of the largest off-diagonal element in
*        column K, and COLMAX is its absolute value.
*        Determine both COLMAX and IMAX.
*
         IF( k.LT.n ) THEN
            imax = k + idamax( n-k, a( k+1, k ), 1 )
            colmax = abs( a( imax, k ) )
         ELSE
            colmax = zero
         END IF
*
         IF( (max( absakk, colmax ).EQ.zero) .OR.
     $       disnan(absakk) ) THEN
*
*           Column K is zero or underflow, or contains a NaN:
*           set INFO and continue
*
            IF( info.EQ.0 )
     $         info = k
            kp = k
         ELSE
            IF( absakk.GE.alpha*colmax ) THEN
*
*              no interchange, use 1-by-1 pivot block
*
               kp = k
            ELSE
*
*              JMAX is the column-index of the largest off-diagonal
*              element in row IMAX, and ROWMAX is its absolute value
*
               jmax = k - 1 + idamax( imax-k, a( imax, k ), lda )
               rowmax = abs( a( imax, jmax ) )
               IF( imax.LT.n ) THEN
                  jmax = imax + idamax( n-imax, a( imax+1, imax ),
     $                                  1 )
                  rowmax = max( rowmax, abs( a( jmax, imax ) ) )
               END IF
*
               IF( absakk.GE.alpha*colmax*( colmax / rowmax ) ) THEN
*
*                 no interchange, use 1-by-1 pivot block
*
                  kp = k
               ELSE IF( abs( a( imax, imax ) ).GE.alpha*rowmax ) THEN
*
*                 interchange rows and columns K and IMAX, use 1-by-1
*                 pivot block
*
                  kp = imax
               ELSE
*
*                 interchange rows and columns K+1 and IMAX, use 2-by-2
*                 pivot block
*
                  kp = imax
                  kstep = 2
                  n2x2 = n2x2 + 1
               END IF
            END IF
*
            kk = k + kstep - 1
            IF( kp.NE.kk ) THEN
*
*              Interchange rows and columns KK and KP in the trailing
*              submatrix A(k:n,k:n)
*
               IF( kp.LT.n )
     $            CALL dswap( n-kp, a( kp+1, kk ), 1, a( kp+1, kp ),
     $                        1 )
               CALL dswap( kp-kk-1, a( kk+1, kk ), 1, a( kp, kk+1 ),
     $                     lda )
               t = a( kk, kk )
               a( kk, kk ) = a( kp, kp )
               a( kp, kp ) = t
               IF( kstep.EQ.2 ) THEN
                  t = a( k+1, k )
                  a( k+1, k ) = a( kp, k )
                  a( kp, k ) = t
               END IF
            END IF
*
*           Update the trailing submatrix
*
            IF( kstep.EQ.1 ) THEN
*
*              1-by-1 pivot block D(k): column k now holds
*
*              W(k) = L(k)*D(k)
*
*              where L(k) is the k-th column of L
*
               IF( k.LT.n ) THEN
*
*                 Perform a rank-1 update of A(k+1:n,k+1:n) as
*
*                 A := A - L(k)*D(k)*L(k)**T = A - W(k)*(1/D(k))*W(k)**T
*
                  d11 = one / a( k, k )
                  CALL dsyr( uplo, n-k, -d11, a( k+1, k ), 1,
     $                       a( k+1, k+1 ), lda )
*
*                 Store L(k) in column K
*
                  CALL dscal( n-k, d11, a( k+1, k ), 1 )
               END IF
            ELSE
*
*              2-by-2 pivot block D(k)
*
               IF( k.LT.n-1 ) THEN
*
*                 Perform a rank-2 update of A(k+2:n,k+2:n) as
*
*                 A := A - ( (A(k) A(k+1))*D(k)**(-1) ) * (A(k) A(k+1))**T
*
*                 where L(k) and L(k+1) are the k-th and (k+1)-th
*                 columns of L
*
                  d21 = a( k+1, k )
                  d11 = a( k+1, k+1 ) / d21
                  d22 = a( k, k ) / d21
                  t = one / ( d11*d22-one )
                  d21 = t / d21
*
                  DO 60 j = k + 2, n
*
                     wk = d21*( d11*a( j, k )-a( j, k+1 ) )
                     wkp1 = d21*( d22*a( j, k+1 )-a( j, k ) )
*
                     DO 50 i = j, n
                        a( i, j ) = a( i, j ) - a( i, k )*wk -
     $                              a( i, k+1 )*wkp1
   50                CONTINUE
*
                     a( j, k ) = wk
                     a( j, k+1 ) = wkp1
*
   60             CONTINUE
               END IF
            END IF
         END IF
*
*        Store details of the interchanges in IPIV
*
         IF( kstep.EQ.1 ) THEN
            ipiv( k ) = kp
         ELSE
            ipiv( k ) = -kp
            ipiv( k+1 ) = -kp
         END IF
*
*        Increase K and return to the start of the main loop
*
         k = k + kstep
         GO TO 40
*
      END IF
*
   70 CONTINUE
*
      RETURN
*
*     End of DSYTF2_MOD
*
      END