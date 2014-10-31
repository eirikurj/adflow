   !        Generated by TAPENADE     (INRIA, Tropics team)
   !  Tapenade 3.10 (r5363) -  9 Sep 2014 09:53
   !
   !  Differentiation of etotarray in reverse (adjoint) mode (with options i4 dr8 r8 noISIZE):
   !   gradient     of useful results: p etot rho (global)tref (global)rgas
   !   with respect to varying inputs: k p u v w etot rho (global)tref
   !                (global)rgas
   !      ==================================================================
   SUBROUTINE ETOTARRAY_B(rho, rhob, u, ub, v, vb, w, wb, p, pb, k, kb, &
   & etot, etotb, correctfork, kk)
   !
   !      ******************************************************************
   !      *                                                                *
   !      * EtotArray computes the total energy from the given density,    *
   !      * velocity and presssure for the given kk elements of the arrays.*
   !      * First the internal energy per unit mass is computed and after  *
   !      * that the kinetic energy is added as well the conversion to     *
   !      * energy per unit volume.                                        *
   !      *                                                                *
   !      ******************************************************************
   !
   USE CONSTANTS
   IMPLICIT NONE
   !
   !      Subroutine arguments.
   !
   REAL(kind=realtype), DIMENSION(*), INTENT(IN) :: rho, p, k
   REAL(kind=realtype), DIMENSION(*) :: rhob, pb, kb
   REAL(kind=realtype), DIMENSION(*), INTENT(IN) :: u, v, w
   REAL(kind=realtype), DIMENSION(*) :: ub, vb, wb
   REAL(kind=realtype), DIMENSION(*) :: etot
   REAL(kind=realtype), DIMENSION(*) :: etotb
   LOGICAL, INTENT(IN) :: correctfork
   INTEGER(kind=inttype), INTENT(IN) :: kk
   !
   !      Local variables.
   !
   INTEGER(kind=inttype) :: i
   REAL(kind=realtype) :: tempb
   !
   !      ******************************************************************
   !      *                                                                *
   !      * Begin execution                                                *
   !      *                                                                *
   !      ******************************************************************
   !
   ! Compute the internal energy for unit mass.
   CALL PUSHREAL8ARRAY(etot, SIZE(etot, 1))
   CALL EINTARRAY(rho, p, k, etot, correctfork, kk)
   ub = 0.0_8
   vb = 0.0_8
   wb = 0.0_8
   DO i=kk,1,-1
   tempb = rho(i)*half*etotb(i)
   rhob(i) = rhob(i) + (etot(i)+half*(u(i)**2+v(i)**2+w(i)**2))*etotb(i&
   &     )
   ub(i) = ub(i) + 2*u(i)*tempb
   vb(i) = vb(i) + 2*v(i)*tempb
   wb(i) = wb(i) + 2*w(i)*tempb
   etotb(i) = rho(i)*etotb(i)
   END DO
   CALL POPREAL8ARRAY(etot, SIZE(etot, 1))
   CALL EINTARRAY_B(rho, rhob, p, pb, k, kb, etot, etotb, correctfork, kk&
   &           )
   END SUBROUTINE ETOTARRAY_B
