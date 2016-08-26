!        generated by tapenade     (inria, tropics team)
!  tapenade 3.10 (r5363) -  9 sep 2014 09:53
!
!  differentiation of residual_block in forward (tangent) mode (with options i4 dr8 r8):
!   variations   of useful results: *dw *w *(*viscsubface.tau)
!   with respect to varying inputs: gammainf timeref rhoinf winf
!                pinfcorr *rev *p *sfacei *sfacej *sfacek *dw *w
!                *rlv *x *vol *si *sj *sk *radi *radj *radk
!   plus diff mem management of: rev:in aa:in wx:in wy:in wz:in
!                p:in sfacei:in sfacej:in sfacek:in dw:in w:in
!                rlv:in x:in qx:in qy:in qz:in ux:in vol:in uy:in
!                uz:in si:in sj:in sk:in vx:in vy:in vz:in fw:in
!                viscsubface:in *viscsubface.tau:in radi:in radj:in
!                radk:in
subroutine residual_block_d()
!
!      ******************************************************************
!      *                                                                *
!      * residual computes the residual of the mean flow equations on   *
!      * the current mg level.                                          *
!      *                                                                *
!      ******************************************************************
!
  use blockpointers
  use cgnsgrid
  use flowvarrefstate
  use inputiteration
  use inputdiscretization
  use inputtimespectral
! added by hdn
  use inputunsteady
  use iteration
  use inputadjoint
  use flowutils_d, only : computespeedofsoundsquared, &
& computespeedofsoundsquared_d
  use diffsizes
!  hint: isize1ofdrfviscsubface should be the size of dimension 1 of array *viscsubface
  implicit none
!
!      local variables.
!
  integer(kind=inttype) :: discr
  integer(kind=inttype) :: i, j, k, l
! for loops of ale
  integer(kind=inttype) :: iale, jale, kale, lale, male
  real(kind=realtype), parameter :: k1=1.05_realtype
! random given number
  real(kind=realtype), parameter :: k2=0.6_realtype
! mach number preconditioner activation
  real(kind=realtype), parameter :: m0=0.2_realtype
  real(kind=realtype), parameter :: alpha=0_realtype
  real(kind=realtype), parameter :: delta=0_realtype
!real(kind=realtype), parameter :: hinf = 2_realtype ! test phase 
! test phase
  real(kind=realtype), parameter :: cpres=4.18_realtype
  real(kind=realtype), parameter :: temp=297.15_realtype
!
!     local variables
!
  real(kind=realtype) :: k3, h, velxrho, velyrho, velzrho, sos, hinf
  real(kind=realtype) :: k3d, velxrhod, velyrhod, velzrhod, sosd
  real(kind=realtype) :: resm, a11, a12, a13, a14, a15, a21, a22, a23, &
& a24, a25, a31, a32, a33, a34, a35
  real(kind=realtype) :: resmd, a11d, a15d, a21d, a22d, a25d, a31d, a33d&
& , a35d
  real(kind=realtype) :: a41, a42, a43, a44, a45, a51, a52, a53, a54, &
& a55, b11, b12, b13, b14, b15
  real(kind=realtype) :: a41d, a44d, a45d, a51d, a52d, a53d, a54d, a55d&
& , b11d, b12d, b13d, b14d, b15d
  real(kind=realtype) :: b21, b22, b23, b24, b25, b31, b32, b33, b34, &
& b35
  real(kind=realtype) :: b21d, b22d, b23d, b24d, b25d, b31d, b32d, b33d&
& , b34d, b35d
  real(kind=realtype) :: b41, b42, b43, b44, b45, b51, b52, b53, b54, &
& b55
  real(kind=realtype) :: b41d, b42d, b43d, b44d, b45d, b51d, b52d, b53d&
& , b54d, b55d
  real(kind=realtype) :: rhohdash, betamr2
  real(kind=realtype) :: betamr2d
  real(kind=realtype) :: g, q
  real(kind=realtype) :: qd
  real(kind=realtype) :: b1, b2, b3, b4, b5
  real(kind=realtype) :: dwo(nwf)
  real(kind=realtype) :: dwod(nwf)
  logical :: finegrid
  intrinsic abs
  intrinsic sqrt
  intrinsic max
  intrinsic min
  intrinsic real
  real(kind=realtype) :: arg1
  real(kind=realtype) :: arg1d
  real(kind=realtype) :: result1
  real(kind=realtype) :: result1d
  real(kind=realtype) :: x3
  real(kind=realtype) :: x2
  real(kind=realtype) :: x1
  real(kind=realtype) :: x1d
  real(kind=realtype) :: abs0
  integer :: ii1
  real(kind=realtype) :: max2
  real(kind=realtype) :: max1
!
!      ******************************************************************
!      *                                                                *
!      * begin execution                                                *
!      *                                                                *
!      ******************************************************************
!
! set the value of rfil, which controls the fraction of the old
! dissipation residual to be used. this is only for the runge-kutta
! schemes; for other smoothers rfil is simply set to 1.0.
! note the index rkstage+1 for cdisrk. the reason is that the
! residual computation is performed before rkstage is incremented.
  if (smoother .eq. rungekutta) then
    rfil = cdisrk(rkstage+1)
  else
    rfil = one
  end if
! initialize the local arrays to monitor the massflows to zero.
! set the value of the discretization, depending on the grid level,
! and the logical finegrid, which indicates whether or not this
! is the finest grid level of the current mg cycle.
  discr = spacediscrcoarse
  if (currentlevel .eq. 1) discr = spacediscr
  finegrid = .false.
  if (currentlevel .eq. groundlevel) finegrid = .true.
! ===========================================================
!
! assuming ale has nothing to do with mg
! the geometric data will be interpolated if in md mode
!
! ===========================================================
! ===========================================================
!
! the fluxes are calculated as usual
!
! ===========================================================
  call inviscidcentralflux_d()
  select case  (discr) 
  case (dissscalar) 
! standard scalar dissipation scheme.
    if (finegrid) then
      if (.not.lumpeddiss) then
        call invisciddissfluxscalar_d()
      else
        call invisciddissfluxscalarapprox_d()
      end if
    else
      fwd = 0.0_8
    end if
  case (dissmatrix) 
!===========================================================
! matrix dissipation scheme.
    if (finegrid) then
      if (.not.lumpeddiss) then
        call invisciddissfluxmatrix_d()
      else
        call invisciddissfluxmatrixapprox_d()
      end if
    else
      fwd = 0.0_8
    end if
  case (upwind) 
!===========================================================
! dissipation via an upwind scheme.
    call inviscidupwindflux_d(finegrid)
  case default
    fwd = 0.0_8
  end select
!-------------------------------------------------------
! lastly, recover the old s[i,j,k], sface[i,j,k]
! this shall be done before difussive and source terms
! are computed.
!-------------------------------------------------------
  if (viscous) then
    if (rfil .ge. 0.) then
      abs0 = rfil
    else
      abs0 = -rfil
    end if
! only compute viscous fluxes if rfil > 0
    if (abs0 .gt. thresholdreal) then
! not lumpeddiss means it isn't the pc...call the vicousflux
      if (.not.lumpeddiss) then
        call computespeedofsoundsquared_d()
        call allnodalgradients_d()
        call viscousflux_d()
      else
! this is a pc calc...only include viscous fluxes if viscpc
! is used
        call computespeedofsoundsquared_d()
        if (viscpc) then
          call allnodalgradients_d()
          call viscousflux_d()
        else
          call viscousfluxapprox_d()
          do ii1=1,isize1ofdrfviscsubface
            viscsubfaced(ii1)%tau = 0.0_8
          end do
        end if
      end if
    else
      do ii1=1,isize1ofdrfviscsubface
        viscsubfaced(ii1)%tau = 0.0_8
      end do
    end if
  else
    do ii1=1,isize1ofdrfviscsubface
      viscsubfaced(ii1)%tau = 0.0_8
    end do
  end if
!===========================================================
! add the dissipative and possibly viscous fluxes to the
! euler fluxes. loop over the owned cells and add fw to dw.
! also multiply by iblank so that no updates occur in holes
  if (lowspeedpreconditioner) then
    dwod = 0.0_8
    do k=2,kl
      do j=2,jl
        do i=2,il
!    compute speed of sound
          arg1d = (gamma(i, j, k)*pd(i, j, k)*w(i, j, k, irho)-gamma(i, &
&           j, k)*p(i, j, k)*wd(i, j, k, irho))/w(i, j, k, irho)**2
          arg1 = gamma(i, j, k)*p(i, j, k)/w(i, j, k, irho)
          if (arg1 .eq. 0.0_8) then
            sosd = 0.0_8
          else
            sosd = arg1d/(2.0*sqrt(arg1))
          end if
          sos = sqrt(arg1)
! coompute velocities without rho from state vector
          velxrhod = wd(i, j, k, ivx)
          velxrho = w(i, j, k, ivx)
          velyrhod = wd(i, j, k, ivy)
          velyrho = w(i, j, k, ivy)
          velzrhod = wd(i, j, k, ivz)
          velzrho = w(i, j, k, ivz)
          qd = 2*velxrho*velxrhod + 2*velyrho*velyrhod + 2*velzrho*&
&           velzrhod
          q = velxrho**2 + velyrho**2 + velzrho**2
          if (q .eq. 0.0_8) then
            result1d = 0.0_8
          else
            result1d = qd/(2.0*sqrt(q))
          end if
          result1 = sqrt(q)
          resmd = (result1d*sos-result1*sosd)/sos**2
          resm = result1/sos
!
!    compute k3
          k3d = (1-k1*m0**2)*2*resm*resmd/m0**4
          k3 = k1*(1+(1-k1*m0**2)*resm**2/(k1*m0**4))
          if (k3*(velxrho**2+velyrho**2+velzrho**2) .lt. k2*(winf(ivx)**&
&             2+winf(ivy)**2+winf(ivz)**2)) then
            x1d = k2*(2*winf(ivx)*winfd(ivx)+2*winf(ivy)*winfd(ivy)+2*&
&             winf(ivz)*winfd(ivz))
            x1 = k2*(winf(ivx)**2+winf(ivy)**2+winf(ivz)**2)
          else
            x1d = k3d*(velxrho**2+velyrho**2+velzrho**2) + k3*(2*velxrho&
&             *velxrhod+2*velyrho*velyrhod+2*velzrho*velzrhod)
            x1 = k3*(velxrho**2+velyrho**2+velzrho**2)
          end if
          if (x1 .gt. sos**2) then
            betamr2d = 2*sos*sosd
            betamr2 = sos**2
          else
            betamr2d = x1d
            betamr2 = x1
          end if
          a11d = betamr2d/sos**4 - betamr2*4*sosd/sos**5
          a11 = betamr2*(1/sos**4)
          a12 = zero
          a13 = zero
          a14 = zero
          a15d = (betamr2*4*sos**3*sosd-betamr2d*sos**4)/(sos**4)**2
          a15 = (-betamr2)/sos**4
          a21d = (one*velxrhod*sos**2-one*velxrho*2*sos*sosd)/(sos**2)**&
&           2
          a21 = one*velxrho/sos**2
          a22d = one*wd(i, j, k, irho)
          a22 = one*w(i, j, k, irho)
          a23 = zero
          a24 = zero
          a25d = (one*velxrho*2*sos*sosd-one*velxrhod*sos**2)/(sos**2)**&
&           2
          a25 = one*(-velxrho)/sos**2
          a31d = (one*velyrhod*sos**2-one*velyrho*2*sos*sosd)/(sos**2)**&
&           2
          a31 = one*velyrho/sos**2
          a32 = zero
          a33d = one*wd(i, j, k, irho)
          a33 = one*w(i, j, k, irho)
          a34 = zero
          a35d = (one*velyrho*2*sos*sosd-one*velyrhod*sos**2)/(sos**2)**&
&           2
          a35 = one*(-velyrho)/sos**2
          a41d = (one*velzrhod*sos**2-one*velzrho*2*sos*sosd)/(sos**2)**&
&           2
          a41 = one*velzrho/sos**2
          a42 = zero
          a43 = zero
          a44d = one*wd(i, j, k, irho)
          a44 = one*w(i, j, k, irho)
          a45d = (one*velzrho*2*sos*sosd-one*velzrhod*sos**2)/(sos**2)**&
&           2
          a45 = zero + one*(-velzrho)/sos**2
          a51d = one*resm*resmd
          a51 = one*(1/(gamma(i, j, k)-1)+resm**2/2)
          a52d = one*(wd(i, j, k, irho)*velxrho+w(i, j, k, irho)*&
&           velxrhod)
          a52 = one*w(i, j, k, irho)*velxrho
          a53d = one*(wd(i, j, k, irho)*velyrho+w(i, j, k, irho)*&
&           velyrhod)
          a53 = one*w(i, j, k, irho)*velyrho
          a54d = one*(wd(i, j, k, irho)*velzrho+w(i, j, k, irho)*&
&           velzrhod)
          a54 = one*w(i, j, k, irho)*velzrho
          a55d = -(one*resm*resmd)
          a55 = one*((-(resm**2))/2)
          b11d = (gamma(i, j, k)-1)*(a11d*q+a11*qd)/2 + (a12*velxrho*wd(&
&           i, j, k, irho)-a12*velxrhod*w(i, j, k, irho))/w(i, j, k, &
&           irho)**2 + (a13*velyrho*wd(i, j, k, irho)-a13*velyrhod*w(i, &
&           j, k, irho))/w(i, j, k, irho)**2 + (a14*velzrho*wd(i, j, k, &
&           irho)-a14*velzrhod*w(i, j, k, irho))/w(i, j, k, irho)**2 + &
&           a15d*((gamma(i, j, k)-1)*q/2-sos**2) + a15*((gamma(i, j, k)-&
&           1)*qd/2-2*sos*sosd)
          b11 = a11*(gamma(i, j, k)-1)*q/2 + a12*(-velxrho)/w(i, j, k, &
&           irho) + a13*(-velyrho)/w(i, j, k, irho) + a14*(-velzrho)/w(i&
&           , j, k, irho) + a15*((gamma(i, j, k)-1)*q/2-sos**2)
          b12d = (1-gamma(i, j, k))*(a11d*velxrho+a11*velxrhod) - a12*wd&
&           (i, j, k, irho)/w(i, j, k, irho)**2 + (1-gamma(i, j, k))*(&
&           a15d*velxrho+a15*velxrhod)
          b12 = a11*(1-gamma(i, j, k))*velxrho + a12*1/w(i, j, k, irho) &
&           + a15*(1-gamma(i, j, k))*velxrho
          b13d = (1-gamma(i, j, k))*(a11d*velyrho+a11*velyrhod) - a13*wd&
&           (i, j, k, irho)/w(i, j, k, irho)**2 + (1-gamma(i, j, k))*(&
&           a15d*velyrho+a15*velyrhod)
          b13 = a11*(1-gamma(i, j, k))*velyrho + a13/w(i, j, k, irho) + &
&           a15*(1-gamma(i, j, k))*velyrho
          b14d = (1-gamma(i, j, k))*(a11d*velzrho+a11*velzrhod) - a14*wd&
&           (i, j, k, irho)/w(i, j, k, irho)**2 + (1-gamma(i, j, k))*(&
&           a15d*velzrho+a15*velzrhod)
          b14 = a11*(1-gamma(i, j, k))*velzrho + a14/w(i, j, k, irho) + &
&           a15*(1-gamma(i, j, k))*velzrho
          b15d = (gamma(i, j, k)-1)*a11d + (gamma(i, j, k)-1)*a15d
          b15 = a11*(gamma(i, j, k)-1) + a15*(gamma(i, j, k)-1)
          b21d = (gamma(i, j, k)-1)*(a21d*q+a21*qd)/2 + ((-(a22d*velxrho&
&           )-a22*velxrhod)*w(i, j, k, irho)+a22*velxrho*wd(i, j, k, &
&           irho))/w(i, j, k, irho)**2 + (a23*velyrho*wd(i, j, k, irho)-&
&           a23*velyrhod*w(i, j, k, irho))/w(i, j, k, irho)**2 + (a24*&
&           velzrho*wd(i, j, k, irho)-a24*velzrhod*w(i, j, k, irho))/w(i&
&           , j, k, irho)**2 + a25d*((gamma(i, j, k)-1)*q/2-sos**2) + &
&           a25*((gamma(i, j, k)-1)*qd/2-2*sos*sosd)
          b21 = a21*(gamma(i, j, k)-1)*q/2 + a22*(-velxrho)/w(i, j, k, &
&           irho) + a23*(-velyrho)/w(i, j, k, irho) + a24*(-velzrho)/w(i&
&           , j, k, irho) + a25*((gamma(i, j, k)-1)*q/2-sos**2)
          b22d = (1-gamma(i, j, k))*(a21d*velxrho+a21*velxrhod) + (a22d*&
&           w(i, j, k, irho)-a22*wd(i, j, k, irho))/w(i, j, k, irho)**2 &
&           + (1-gamma(i, j, k))*(a25d*velxrho+a25*velxrhod)
          b22 = a21*(1-gamma(i, j, k))*velxrho + a22/w(i, j, k, irho) + &
&           a25*(1-gamma(i, j, k))*velxrho
          b23d = (1-gamma(i, j, k))*(a21d*velyrho+a21*velyrhod) - a23*wd&
&           (i, j, k, irho)/w(i, j, k, irho)**2 + (1-gamma(i, j, k))*(&
&           a25d*velyrho+a25*velyrhod)
          b23 = a21*(1-gamma(i, j, k))*velyrho + a23*1/w(i, j, k, irho) &
&           + a25*(1-gamma(i, j, k))*velyrho
          b24d = (1-gamma(i, j, k))*(a21d*velzrho+a21*velzrhod) - a24*wd&
&           (i, j, k, irho)/w(i, j, k, irho)**2 + (1-gamma(i, j, k))*(&
&           a25d*velzrho+a25*velzrhod)
          b24 = a21*(1-gamma(i, j, k))*velzrho + a24*1/w(i, j, k, irho) &
&           + a25*(1-gamma(i, j, k))*velzrho
          b25d = (gamma(i, j, k)-1)*a21d + (gamma(i, j, k)-1)*a25d
          b25 = a21*(gamma(i, j, k)-1) + a25*(gamma(i, j, k)-1)
          b31d = (gamma(i, j, k)-1)*(a31d*q+a31*qd)/2 + (a32*velxrho*wd(&
&           i, j, k, irho)-a32*velxrhod*w(i, j, k, irho))/w(i, j, k, &
&           irho)**2 + ((-(a33d*velyrho)-a33*velyrhod)*w(i, j, k, irho)+&
&           a33*velyrho*wd(i, j, k, irho))/w(i, j, k, irho)**2 + (a34*&
&           velzrho*wd(i, j, k, irho)-a34*velzrhod*w(i, j, k, irho))/w(i&
&           , j, k, irho)**2 + a35d*((gamma(i, j, k)-1)*q/2-sos**2) + &
&           a35*((gamma(i, j, k)-1)*qd/2-2*sos*sosd)
          b31 = a31*(gamma(i, j, k)-1)*q/2 + a32*(-velxrho)/w(i, j, k, &
&           irho) + a33*(-velyrho)/w(i, j, k, irho) + a34*(-velzrho)/w(i&
&           , j, k, irho) + a35*((gamma(i, j, k)-1)*q/2-sos**2)
          b32d = (1-gamma(i, j, k))*(a31d*velxrho+a31*velxrhod) - a32*wd&
&           (i, j, k, irho)/w(i, j, k, irho)**2 + (1-gamma(i, j, k))*(&
&           a35d*velxrho+a35*velxrhod)
          b32 = a31*(1-gamma(i, j, k))*velxrho + a32/w(i, j, k, irho) + &
&           a35*(1-gamma(i, j, k))*velxrho
          b33d = (1-gamma(i, j, k))*(a31d*velyrho+a31*velyrhod) + (a33d*&
&           w(i, j, k, irho)-a33*wd(i, j, k, irho))/w(i, j, k, irho)**2 &
&           + (1-gamma(i, j, k))*(a35d*velyrho+a35*velyrhod)
          b33 = a31*(1-gamma(i, j, k))*velyrho + a33*1/w(i, j, k, irho) &
&           + a35*(1-gamma(i, j, k))*velyrho
          b34d = (1-gamma(i, j, k))*(a31d*velzrho+a31*velzrhod) - a34*wd&
&           (i, j, k, irho)/w(i, j, k, irho)**2 + (1-gamma(i, j, k))*(&
&           a35d*velzrho+a35*velzrhod)
          b34 = a31*(1-gamma(i, j, k))*velzrho + a34*1/w(i, j, k, irho) &
&           + a35*(1-gamma(i, j, k))*velzrho
          b35d = (gamma(i, j, k)-1)*a31d + (gamma(i, j, k)-1)*a35d
          b35 = a31*(gamma(i, j, k)-1) + a35*(gamma(i, j, k)-1)
          b41d = (gamma(i, j, k)-1)*(a41d*q+a41*qd)/2 + (a42*velxrho*wd(&
&           i, j, k, irho)-a42*velxrhod*w(i, j, k, irho))/w(i, j, k, &
&           irho)**2 + (a43*velyrho*wd(i, j, k, irho)-a43*velyrhod*w(i, &
&           j, k, irho))/w(i, j, k, irho)**2 + ((-(a44d*velzrho)-a44*&
&           velzrhod)*w(i, j, k, irho)+a44*velzrho*wd(i, j, k, irho))/w(&
&           i, j, k, irho)**2 + a45d*((gamma(i, j, k)-1)*q/2-sos**2) + &
&           a45*((gamma(i, j, k)-1)*qd/2-2*sos*sosd)
          b41 = a41*(gamma(i, j, k)-1)*q/2 + a42*(-velxrho)/w(i, j, k, &
&           irho) + a43*(-velyrho)/w(i, j, k, irho) + a44*(-velzrho)/w(i&
&           , j, k, irho) + a45*((gamma(i, j, k)-1)*q/2-sos**2)
          b42d = (1-gamma(i, j, k))*(a41d*velxrho+a41*velxrhod) - a42*wd&
&           (i, j, k, irho)/w(i, j, k, irho)**2 + (1-gamma(i, j, k))*(&
&           a45d*velxrho+a45*velxrhod)
          b42 = a41*(1-gamma(i, j, k))*velxrho + a42/w(i, j, k, irho) + &
&           a45*(1-gamma(i, j, k))*velxrho
          b43d = (1-gamma(i, j, k))*(a41d*velyrho+a41*velyrhod) - a43*wd&
&           (i, j, k, irho)/w(i, j, k, irho)**2 + (1-gamma(i, j, k))*(&
&           a45d*velyrho+a45*velyrhod)
          b43 = a41*(1-gamma(i, j, k))*velyrho + a43*1/w(i, j, k, irho) &
&           + a45*(1-gamma(i, j, k))*velyrho
          b44d = (1-gamma(i, j, k))*(a41d*velzrho+a41*velzrhod) + (a44d*&
&           w(i, j, k, irho)-a44*wd(i, j, k, irho))/w(i, j, k, irho)**2 &
&           + (1-gamma(i, j, k))*(a45d*velzrho+a45*velzrhod)
          b44 = a41*(1-gamma(i, j, k))*velzrho + a44*1/w(i, j, k, irho) &
&           + a45*(1-gamma(i, j, k))*velzrho
          b45d = (gamma(i, j, k)-1)*a41d + (gamma(i, j, k)-1)*a45d
          b45 = a41*(gamma(i, j, k)-1) + a45*(gamma(i, j, k)-1)
          b51d = (gamma(i, j, k)-1)*(a51d*q+a51*qd)/2 + ((-(a52d*velxrho&
&           )-a52*velxrhod)*w(i, j, k, irho)+a52*velxrho*wd(i, j, k, &
&           irho))/w(i, j, k, irho)**2 + ((-(a53d*velyrho)-a53*velyrhod)&
&           *w(i, j, k, irho)+a53*velyrho*wd(i, j, k, irho))/w(i, j, k, &
&           irho)**2 + ((-(a54d*velzrho)-a54*velzrhod)*w(i, j, k, irho)+&
&           a54*velzrho*wd(i, j, k, irho))/w(i, j, k, irho)**2 + a55d*((&
&           gamma(i, j, k)-1)*q/2-sos**2) + a55*((gamma(i, j, k)-1)*qd/2&
&           -2*sos*sosd)
          b51 = a51*(gamma(i, j, k)-1)*q/2 + a52*(-velxrho)/w(i, j, k, &
&           irho) + a53*(-velyrho)/w(i, j, k, irho) + a54*(-velzrho)/w(i&
&           , j, k, irho) + a55*((gamma(i, j, k)-1)*q/2-sos**2)
          b52d = (1-gamma(i, j, k))*(a51d*velxrho+a51*velxrhod) + (a52d*&
&           w(i, j, k, irho)-a52*wd(i, j, k, irho))/w(i, j, k, irho)**2 &
&           + (1-gamma(i, j, k))*(a55d*velxrho+a55*velxrhod)
          b52 = a51*(1-gamma(i, j, k))*velxrho + a52/w(i, j, k, irho) + &
&           a55*(1-gamma(i, j, k))*velxrho
          b53d = (1-gamma(i, j, k))*(a51d*velyrho+a51*velyrhod) + (a53d*&
&           w(i, j, k, irho)-a53*wd(i, j, k, irho))/w(i, j, k, irho)**2 &
&           + (1-gamma(i, j, k))*(a55d*velyrho+a55*velyrhod)
          b53 = a51*(1-gamma(i, j, k))*velyrho + a53*1/w(i, j, k, irho) &
&           + a55*(1-gamma(i, j, k))*velyrho
          b54d = (1-gamma(i, j, k))*(a51d*velzrho+a51*velzrhod) + (a54d*&
&           w(i, j, k, irho)-a54*wd(i, j, k, irho))/w(i, j, k, irho)**2 &
&           + (1-gamma(i, j, k))*(a55d*velzrho+a55*velzrhod)
          b54 = a51*(1-gamma(i, j, k))*velzrho + a54*1/w(i, j, k, irho) &
&           + a55*(1-gamma(i, j, k))*velzrho
          b55d = (gamma(i, j, k)-1)*a51d + (gamma(i, j, k)-1)*a55d
          b55 = a51*(gamma(i, j, k)-1) + a55*(gamma(i, j, k)-1)
! dwo is the orginal redisual
          do l=1,nwf
            x2 = real(iblank(i, j, k), realtype)
            if (x2 .lt. zero) then
              max1 = zero
            else
              max1 = x2
            end if
            dwod(l) = max1*(dwd(i, j, k, l)+fwd(i, j, k, l))
            dwo(l) = (dw(i, j, k, l)+fw(i, j, k, l))*max1
          end do
          dwd(i, j, k, 1) = b11d*dwo(1) + b11*dwod(1) + b12d*dwo(2) + &
&           b12*dwod(2) + b13d*dwo(3) + b13*dwod(3) + b14d*dwo(4) + b14*&
&           dwod(4) + b15d*dwo(5) + b15*dwod(5)
          dw(i, j, k, 1) = b11*dwo(1) + b12*dwo(2) + b13*dwo(3) + b14*&
&           dwo(4) + b15*dwo(5)
          dwd(i, j, k, 2) = b21d*dwo(1) + b21*dwod(1) + b22d*dwo(2) + &
&           b22*dwod(2) + b23d*dwo(3) + b23*dwod(3) + b24d*dwo(4) + b24*&
&           dwod(4) + b25d*dwo(5) + b25*dwod(5)
          dw(i, j, k, 2) = b21*dwo(1) + b22*dwo(2) + b23*dwo(3) + b24*&
&           dwo(4) + b25*dwo(5)
          dwd(i, j, k, 3) = b31d*dwo(1) + b31*dwod(1) + b32d*dwo(2) + &
&           b32*dwod(2) + b33d*dwo(3) + b33*dwod(3) + b34d*dwo(4) + b34*&
&           dwod(4) + b35d*dwo(5) + b35*dwod(5)
          dw(i, j, k, 3) = b31*dwo(1) + b32*dwo(2) + b33*dwo(3) + b34*&
&           dwo(4) + b35*dwo(5)
          dwd(i, j, k, 4) = b41d*dwo(1) + b41*dwod(1) + b42d*dwo(2) + &
&           b42*dwod(2) + b43d*dwo(3) + b43*dwod(3) + b44d*dwo(4) + b44*&
&           dwod(4) + b45d*dwo(5) + b45*dwod(5)
          dw(i, j, k, 4) = b41*dwo(1) + b42*dwo(2) + b43*dwo(3) + b44*&
&           dwo(4) + b45*dwo(5)
          dwd(i, j, k, 5) = b51d*dwo(1) + b51*dwod(1) + b52d*dwo(2) + &
&           b52*dwod(2) + b53d*dwo(3) + b53*dwod(3) + b54d*dwo(4) + b54*&
&           dwod(4) + b55d*dwo(5) + b55*dwod(5)
          dw(i, j, k, 5) = b51*dwo(1) + b52*dwo(2) + b53*dwo(3) + b54*&
&           dwo(4) + b55*dwo(5)
        end do
      end do
    end do
  else
    do l=1,nwf
      do k=2,kl
        do j=2,jl
          do i=2,il
            x3 = real(iblank(i, j, k), realtype)
            if (x3 .lt. zero) then
              max2 = zero
            else
              max2 = x3
            end if
            dwd(i, j, k, l) = max2*(dwd(i, j, k, l)+fwd(i, j, k, l))
            dw(i, j, k, l) = (dw(i, j, k, l)+fw(i, j, k, l))*max2
          end do
        end do
      end do
    end do
  end if
end subroutine residual_block_d