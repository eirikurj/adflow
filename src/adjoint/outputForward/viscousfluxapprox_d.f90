!        generated by tapenade     (inria, tropics team)
!  tapenade 3.10 (r5363) -  9 sep 2014 09:53
!
!  differentiation of viscousfluxapprox in forward (tangent) mode (with options i4 dr8 r8):
!   variations   of useful results: *fw
!   with respect to varying inputs: *rev *aa *w *rlv *x *si *sj
!                *sk *fw
!   plus diff mem management of: rev:in aa:in w:in rlv:in x:in
!                si:in sj:in sk:in fw:in
subroutine viscousfluxapprox_d()
  use constants
  use blockpointers
  use flowvarrefstate
  use inputphysics
  use iteration
  implicit none
!
!      local parameter.
!
  real(kind=realtype), parameter :: twothird=two*third
!
!      local variables.
!
  integer(kind=inttype) :: i, j, k
  integer(kind=inttype) :: ii, jj, kk
  real(kind=realtype) :: rfilv, por, mul, mue, mut, heatcoef
  real(kind=realtype) :: muld, mued, mutd, heatcoefd
  real(kind=realtype) :: gm1, factlamheat, factturbheat
  real(kind=realtype) :: u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z
  real(kind=realtype) :: u_xd, u_yd, u_zd, v_xd, v_yd, v_zd, w_xd, w_yd&
& , w_zd
  real(kind=realtype) :: q_x, q_y, q_z, ubar, vbar, wbar
  real(kind=realtype) :: q_xd, q_yd, q_zd, ubard, vbard, wbard
  real(kind=realtype) :: corr, ssx, ssy, ssz, ss, fracdiv
  real(kind=realtype) :: ssxd, ssyd, sszd, ssd, fracdivd
  real(kind=realtype) :: tauxx, tauyy, tauzz
  real(kind=realtype) :: tauxxd, tauyyd, tauzzd
  real(kind=realtype) :: tauxy, tauxz, tauyz
  real(kind=realtype) :: tauxyd, tauxzd, tauyzd
  real(kind=realtype) :: fmx, fmy, fmz, frhoe
  real(kind=realtype) :: fmxd, fmyd, fmzd, frhoed
  real(kind=realtype) :: dd
  real(kind=realtype) :: ddd
  logical :: correctfork
  mue = zero
  rfilv = rfil
  mued = 0.0_8
! viscous fluxes in the i-direction
  do k=2,kl
    do j=2,jl
      do i=1,il
! compute the vector from the center of cell i to cell i+1           
        ssxd = eighth*(xd(i+1, j-1, k-1, 1)-xd(i-1, j-1, k-1, 1)+xd(i+1&
&         , j-1, k, 1)-xd(i-1, j-1, k, 1)+xd(i+1, j, k-1, 1)-xd(i-1, j, &
&         k-1, 1)+xd(i+1, j, k, 1)-xd(i-1, j, k, 1))
        ssx = eighth*(x(i+1, j-1, k-1, 1)-x(i-1, j-1, k-1, 1)+x(i+1, j-1&
&         , k, 1)-x(i-1, j-1, k, 1)+x(i+1, j, k-1, 1)-x(i-1, j, k-1, 1)+&
&         x(i+1, j, k, 1)-x(i-1, j, k, 1))
        ssyd = eighth*(xd(i+1, j-1, k-1, 2)-xd(i-1, j-1, k-1, 2)+xd(i+1&
&         , j-1, k, 2)-xd(i-1, j-1, k, 2)+xd(i+1, j, k-1, 2)-xd(i-1, j, &
&         k-1, 2)+xd(i+1, j, k, 2)-xd(i-1, j, k, 2))
        ssy = eighth*(x(i+1, j-1, k-1, 2)-x(i-1, j-1, k-1, 2)+x(i+1, j-1&
&         , k, 2)-x(i-1, j-1, k, 2)+x(i+1, j, k-1, 2)-x(i-1, j, k-1, 2)+&
&         x(i+1, j, k, 2)-x(i-1, j, k, 2))
        sszd = eighth*(xd(i+1, j-1, k-1, 3)-xd(i-1, j-1, k-1, 3)+xd(i+1&
&         , j-1, k, 3)-xd(i-1, j-1, k, 3)+xd(i+1, j, k-1, 3)-xd(i-1, j, &
&         k-1, 3)+xd(i+1, j, k, 3)-xd(i-1, j, k, 3))
        ssz = eighth*(x(i+1, j-1, k-1, 3)-x(i-1, j-1, k-1, 3)+x(i+1, j-1&
&         , k, 3)-x(i-1, j-1, k, 3)+x(i+1, j, k-1, 3)-x(i-1, j, k-1, 3)+&
&         x(i+1, j, k, 3)-x(i-1, j, k, 3))
! and determine one/ length of vector squared
        ssd = -(one*(ssxd*ssx+ssx*ssxd+ssyd*ssy+ssy*ssyd+sszd*ssz+ssz*&
&         sszd)/(ssx*ssx+ssy*ssy+ssz*ssz)**2)
        ss = one/(ssx*ssx+ssy*ssy+ssz*ssz)
        ssxd = ssd*ssx + ss*ssxd
        ssx = ss*ssx
        ssyd = ssd*ssy + ss*ssyd
        ssy = ss*ssy
        sszd = ssd*ssz + ss*sszd
        ssz = ss*ssz
! now compute each gradient
        ddd = wd(i+1, j, k, ivx) - wd(i, j, k, ivx)
        dd = w(i+1, j, k, ivx) - w(i, j, k, ivx)
        u_xd = ddd*ssx + dd*ssxd
        u_x = dd*ssx
        u_yd = ddd*ssy + dd*ssyd
        u_y = dd*ssy
        u_zd = ddd*ssz + dd*sszd
        u_z = dd*ssz
        ddd = wd(i+1, j, k, ivy) - wd(i, j, k, ivy)
        dd = w(i+1, j, k, ivy) - w(i, j, k, ivy)
        v_xd = ddd*ssx + dd*ssxd
        v_x = dd*ssx
        v_yd = ddd*ssy + dd*ssyd
        v_y = dd*ssy
        v_zd = ddd*ssz + dd*sszd
        v_z = dd*ssz
        ddd = wd(i+1, j, k, ivz) - wd(i, j, k, ivz)
        dd = w(i+1, j, k, ivz) - w(i, j, k, ivz)
        w_xd = ddd*ssx + dd*ssxd
        w_x = dd*ssx
        w_yd = ddd*ssy + dd*ssyd
        w_y = dd*ssy
        w_zd = ddd*ssz + dd*sszd
        w_z = dd*ssz
        ddd = aad(i+1, j, k) - aad(i, j, k)
        dd = aa(i+1, j, k) - aa(i, j, k)
        q_xd = -(ddd*ssx+dd*ssxd)
        q_x = -(dd*ssx)
        q_yd = -(ddd*ssy+dd*ssyd)
        q_y = -(dd*ssy)
        q_zd = -(ddd*ssz+dd*sszd)
        q_z = -(dd*ssz)
        por = half*rfilv
        if (pori(i, j, k) .eq. noflux) por = zero
! compute the laminar and (if present) the eddy viscosities
! multiplied by the porosity. compute the factor in front of
! the gradients of the speed of sound squared for the heat
! flux.
        muld = por*(rlvd(i, j, k)+rlvd(i+1, j, k))
        mul = por*(rlv(i, j, k)+rlv(i+1, j, k))
        if (eddymodel) then
          mued = por*(revd(i, j, k)+revd(i+1, j, k))
          mue = por*(rev(i, j, k)+rev(i+1, j, k))
        end if
        mutd = muld + mued
        mut = mul + mue
        gm1 = half*(gamma(i, j, k)+gamma(i+1, j, k)) - one
        factlamheat = one/(prandtl*gm1)
        factturbheat = one/(prandtlturb*gm1)
        heatcoefd = factlamheat*muld + factturbheat*mued
        heatcoef = mul*factlamheat + mue*factturbheat
! compute the stress tensor and the heat flux vector.
        fracdivd = twothird*(u_xd+v_yd+w_zd)
        fracdiv = twothird*(u_x+v_y+w_z)
        tauxxd = mutd*(two*u_x-fracdiv) + mut*(two*u_xd-fracdivd)
        tauxx = mut*(two*u_x-fracdiv)
        tauyyd = mutd*(two*v_y-fracdiv) + mut*(two*v_yd-fracdivd)
        tauyy = mut*(two*v_y-fracdiv)
        tauzzd = mutd*(two*w_z-fracdiv) + mut*(two*w_zd-fracdivd)
        tauzz = mut*(two*w_z-fracdiv)
        tauxyd = mutd*(u_y+v_x) + mut*(u_yd+v_xd)
        tauxy = mut*(u_y+v_x)
        tauxzd = mutd*(u_z+w_x) + mut*(u_zd+w_xd)
        tauxz = mut*(u_z+w_x)
        tauyzd = mutd*(v_z+w_y) + mut*(v_zd+w_yd)
        tauyz = mut*(v_z+w_y)
        q_xd = heatcoefd*q_x + heatcoef*q_xd
        q_x = heatcoef*q_x
        q_yd = heatcoefd*q_y + heatcoef*q_yd
        q_y = heatcoef*q_y
        q_zd = heatcoefd*q_z + heatcoef*q_zd
        q_z = heatcoef*q_z
! compute the average velocities for the face. remember that
! the velocities are stored and not the momentum.
        ubard = half*(wd(i, j, k, ivx)+wd(i+1, j, k, ivx))
        ubar = half*(w(i, j, k, ivx)+w(i+1, j, k, ivx))
        vbard = half*(wd(i, j, k, ivy)+wd(i+1, j, k, ivy))
        vbar = half*(w(i, j, k, ivy)+w(i+1, j, k, ivy))
        wbard = half*(wd(i, j, k, ivz)+wd(i+1, j, k, ivz))
        wbar = half*(w(i, j, k, ivz)+w(i+1, j, k, ivz))
! compute the viscous fluxes for this i-face.
        fmxd = tauxxd*si(i, j, k, 1) + tauxx*sid(i, j, k, 1) + tauxyd*si&
&         (i, j, k, 2) + tauxy*sid(i, j, k, 2) + tauxzd*si(i, j, k, 3) +&
&         tauxz*sid(i, j, k, 3)
        fmx = tauxx*si(i, j, k, 1) + tauxy*si(i, j, k, 2) + tauxz*si(i, &
&         j, k, 3)
        fmyd = tauxyd*si(i, j, k, 1) + tauxy*sid(i, j, k, 1) + tauyyd*si&
&         (i, j, k, 2) + tauyy*sid(i, j, k, 2) + tauyzd*si(i, j, k, 3) +&
&         tauyz*sid(i, j, k, 3)
        fmy = tauxy*si(i, j, k, 1) + tauyy*si(i, j, k, 2) + tauyz*si(i, &
&         j, k, 3)
        fmzd = tauxzd*si(i, j, k, 1) + tauxz*sid(i, j, k, 1) + tauyzd*si&
&         (i, j, k, 2) + tauyz*sid(i, j, k, 2) + tauzzd*si(i, j, k, 3) +&
&         tauzz*sid(i, j, k, 3)
        fmz = tauxz*si(i, j, k, 1) + tauyz*si(i, j, k, 2) + tauzz*si(i, &
&         j, k, 3)
        frhoed = (ubard*tauxx+ubar*tauxxd+vbard*tauxy+vbar*tauxyd+wbard*&
&         tauxz+wbar*tauxzd)*si(i, j, k, 1) + (ubar*tauxx+vbar*tauxy+&
&         wbar*tauxz)*sid(i, j, k, 1) + (ubard*tauxy+ubar*tauxyd+vbard*&
&         tauyy+vbar*tauyyd+wbard*tauyz+wbar*tauyzd)*si(i, j, k, 2) + (&
&         ubar*tauxy+vbar*tauyy+wbar*tauyz)*sid(i, j, k, 2) + (ubard*&
&         tauxz+ubar*tauxzd+vbard*tauyz+vbar*tauyzd+wbard*tauzz+wbar*&
&         tauzzd)*si(i, j, k, 3) + (ubar*tauxz+vbar*tauyz+wbar*tauzz)*&
&         sid(i, j, k, 3) - q_xd*si(i, j, k, 1) - q_x*sid(i, j, k, 1) - &
&         q_yd*si(i, j, k, 2) - q_y*sid(i, j, k, 2) - q_zd*si(i, j, k, 3&
&         ) - q_z*sid(i, j, k, 3)
        frhoe = (ubar*tauxx+vbar*tauxy+wbar*tauxz)*si(i, j, k, 1) + (&
&         ubar*tauxy+vbar*tauyy+wbar*tauyz)*si(i, j, k, 2) + (ubar*tauxz&
&         +vbar*tauyz+wbar*tauzz)*si(i, j, k, 3) - q_x*si(i, j, k, 1) - &
&         q_y*si(i, j, k, 2) - q_z*si(i, j, k, 3)
! update the residuals of cell i and i+1.
        fwd(i, j, k, imx) = fwd(i, j, k, imx) - fmxd
        fw(i, j, k, imx) = fw(i, j, k, imx) - fmx
        fwd(i, j, k, imy) = fwd(i, j, k, imy) - fmyd
        fw(i, j, k, imy) = fw(i, j, k, imy) - fmy
        fwd(i, j, k, imz) = fwd(i, j, k, imz) - fmzd
        fw(i, j, k, imz) = fw(i, j, k, imz) - fmz
        fwd(i, j, k, irhoe) = fwd(i, j, k, irhoe) - frhoed
        fw(i, j, k, irhoe) = fw(i, j, k, irhoe) - frhoe
        fwd(i+1, j, k, imx) = fwd(i+1, j, k, imx) + fmxd
        fw(i+1, j, k, imx) = fw(i+1, j, k, imx) + fmx
        fwd(i+1, j, k, imy) = fwd(i+1, j, k, imy) + fmyd
        fw(i+1, j, k, imy) = fw(i+1, j, k, imy) + fmy
        fwd(i+1, j, k, imz) = fwd(i+1, j, k, imz) + fmzd
        fw(i+1, j, k, imz) = fw(i+1, j, k, imz) + fmz
        fwd(i+1, j, k, irhoe) = fwd(i+1, j, k, irhoe) + frhoed
        fw(i+1, j, k, irhoe) = fw(i+1, j, k, irhoe) + frhoe
      end do
    end do
  end do
! viscous fluxes in the j-direction
  do k=2,kl
    do j=1,jl
      do i=2,il
! compute the vector from the center of cell j to cell j+1           
        ssxd = eighth*(xd(i-1, j+1, k-1, 1)-xd(i-1, j-1, k-1, 1)+xd(i-1&
&         , j+1, k, 1)-xd(i-1, j-1, k, 1)+xd(i, j+1, k-1, 1)-xd(i, j-1, &
&         k-1, 1)+xd(i, j+1, k, 1)-xd(i, j-1, k, 1))
        ssx = eighth*(x(i-1, j+1, k-1, 1)-x(i-1, j-1, k-1, 1)+x(i-1, j+1&
&         , k, 1)-x(i-1, j-1, k, 1)+x(i, j+1, k-1, 1)-x(i, j-1, k-1, 1)+&
&         x(i, j+1, k, 1)-x(i, j-1, k, 1))
        ssyd = eighth*(xd(i-1, j+1, k-1, 2)-xd(i-1, j-1, k-1, 2)+xd(i-1&
&         , j+1, k, 2)-xd(i-1, j-1, k, 2)+xd(i, j+1, k-1, 2)-xd(i, j-1, &
&         k-1, 2)+xd(i, j+1, k, 2)-xd(i, j-1, k, 2))
        ssy = eighth*(x(i-1, j+1, k-1, 2)-x(i-1, j-1, k-1, 2)+x(i-1, j+1&
&         , k, 2)-x(i-1, j-1, k, 2)+x(i, j+1, k-1, 2)-x(i, j-1, k-1, 2)+&
&         x(i, j+1, k, 2)-x(i, j-1, k, 2))
        sszd = eighth*(xd(i-1, j+1, k-1, 3)-xd(i-1, j-1, k-1, 3)+xd(i-1&
&         , j+1, k, 3)-xd(i-1, j-1, k, 3)+xd(i, j+1, k-1, 3)-xd(i, j-1, &
&         k-1, 3)+xd(i, j+1, k, 3)-xd(i, j-1, k, 3))
        ssz = eighth*(x(i-1, j+1, k-1, 3)-x(i-1, j-1, k-1, 3)+x(i-1, j+1&
&         , k, 3)-x(i-1, j-1, k, 3)+x(i, j+1, k-1, 3)-x(i, j-1, k-1, 3)+&
&         x(i, j+1, k, 3)-x(i, j-1, k, 3))
! and determine one/ length of vector squared
        ssd = -(one*(ssxd*ssx+ssx*ssxd+ssyd*ssy+ssy*ssyd+sszd*ssz+ssz*&
&         sszd)/(ssx*ssx+ssy*ssy+ssz*ssz)**2)
        ss = one/(ssx*ssx+ssy*ssy+ssz*ssz)
        ssxd = ssd*ssx + ss*ssxd
        ssx = ss*ssx
        ssyd = ssd*ssy + ss*ssyd
        ssy = ss*ssy
        sszd = ssd*ssz + ss*sszd
        ssz = ss*ssz
! now compute each gradient
        ddd = wd(i, j+1, k, ivx) - wd(i, j, k, ivx)
        dd = w(i, j+1, k, ivx) - w(i, j, k, ivx)
        u_xd = ddd*ssx + dd*ssxd
        u_x = dd*ssx
        u_yd = ddd*ssy + dd*ssyd
        u_y = dd*ssy
        u_zd = ddd*ssz + dd*sszd
        u_z = dd*ssz
        ddd = wd(i, j+1, k, ivy) - wd(i, j, k, ivy)
        dd = w(i, j+1, k, ivy) - w(i, j, k, ivy)
        v_xd = ddd*ssx + dd*ssxd
        v_x = dd*ssx
        v_yd = ddd*ssy + dd*ssyd
        v_y = dd*ssy
        v_zd = ddd*ssz + dd*sszd
        v_z = dd*ssz
        ddd = wd(i, j+1, k, ivz) - wd(i, j, k, ivz)
        dd = w(i, j+1, k, ivz) - w(i, j, k, ivz)
        w_xd = ddd*ssx + dd*ssxd
        w_x = dd*ssx
        w_yd = ddd*ssy + dd*ssyd
        w_y = dd*ssy
        w_zd = ddd*ssz + dd*sszd
        w_z = dd*ssz
        ddd = aad(i, j+1, k) - aad(i, j, k)
        dd = aa(i, j+1, k) - aa(i, j, k)
        q_xd = -(ddd*ssx+dd*ssxd)
        q_x = -(dd*ssx)
        q_yd = -(ddd*ssy+dd*ssyd)
        q_y = -(dd*ssy)
        q_zd = -(ddd*ssz+dd*sszd)
        q_z = -(dd*ssz)
        por = half*rfilv
        if (porj(i, j, k) .eq. noflux) por = zero
! compute the laminar and (if present) the eddy viscosities
! multiplied by the porosity. compute the factor in front of
! the gradients of the speed of sound squared for the heat
! flux.
        muld = por*(rlvd(i, j, k)+rlvd(i, j+1, k))
        mul = por*(rlv(i, j, k)+rlv(i, j+1, k))
        if (eddymodel) then
          mued = por*(revd(i, j, k)+revd(i, j+1, k))
          mue = por*(rev(i, j, k)+rev(i, j+1, k))
        end if
        mutd = muld + mued
        mut = mul + mue
        gm1 = half*(gamma(i, j, k)+gamma(i, j+1, k)) - one
        factlamheat = one/(prandtl*gm1)
        factturbheat = one/(prandtlturb*gm1)
        heatcoefd = factlamheat*muld + factturbheat*mued
        heatcoef = mul*factlamheat + mue*factturbheat
! compute the stress tensor and the heat flux vector.
        fracdivd = twothird*(u_xd+v_yd+w_zd)
        fracdiv = twothird*(u_x+v_y+w_z)
        tauxxd = mutd*(two*u_x-fracdiv) + mut*(two*u_xd-fracdivd)
        tauxx = mut*(two*u_x-fracdiv)
        tauyyd = mutd*(two*v_y-fracdiv) + mut*(two*v_yd-fracdivd)
        tauyy = mut*(two*v_y-fracdiv)
        tauzzd = mutd*(two*w_z-fracdiv) + mut*(two*w_zd-fracdivd)
        tauzz = mut*(two*w_z-fracdiv)
        tauxyd = mutd*(u_y+v_x) + mut*(u_yd+v_xd)
        tauxy = mut*(u_y+v_x)
        tauxzd = mutd*(u_z+w_x) + mut*(u_zd+w_xd)
        tauxz = mut*(u_z+w_x)
        tauyzd = mutd*(v_z+w_y) + mut*(v_zd+w_yd)
        tauyz = mut*(v_z+w_y)
        q_xd = heatcoefd*q_x + heatcoef*q_xd
        q_x = heatcoef*q_x
        q_yd = heatcoefd*q_y + heatcoef*q_yd
        q_y = heatcoef*q_y
        q_zd = heatcoefd*q_z + heatcoef*q_zd
        q_z = heatcoef*q_z
! compute the average velocities for the face. remember that
! the velocities are stored and not the momentum.
        ubard = half*(wd(i, j, k, ivx)+wd(i, j+1, k, ivx))
        ubar = half*(w(i, j, k, ivx)+w(i, j+1, k, ivx))
        vbard = half*(wd(i, j, k, ivy)+wd(i, j+1, k, ivy))
        vbar = half*(w(i, j, k, ivy)+w(i, j+1, k, ivy))
        wbard = half*(wd(i, j, k, ivz)+wd(i, j+1, k, ivz))
        wbar = half*(w(i, j, k, ivz)+w(i, j+1, k, ivz))
! compute the viscous fluxes for this j-face.
        fmxd = tauxxd*sj(i, j, k, 1) + tauxx*sjd(i, j, k, 1) + tauxyd*sj&
&         (i, j, k, 2) + tauxy*sjd(i, j, k, 2) + tauxzd*sj(i, j, k, 3) +&
&         tauxz*sjd(i, j, k, 3)
        fmx = tauxx*sj(i, j, k, 1) + tauxy*sj(i, j, k, 2) + tauxz*sj(i, &
&         j, k, 3)
        fmyd = tauxyd*sj(i, j, k, 1) + tauxy*sjd(i, j, k, 1) + tauyyd*sj&
&         (i, j, k, 2) + tauyy*sjd(i, j, k, 2) + tauyzd*sj(i, j, k, 3) +&
&         tauyz*sjd(i, j, k, 3)
        fmy = tauxy*sj(i, j, k, 1) + tauyy*sj(i, j, k, 2) + tauyz*sj(i, &
&         j, k, 3)
        fmzd = tauxzd*sj(i, j, k, 1) + tauxz*sjd(i, j, k, 1) + tauyzd*sj&
&         (i, j, k, 2) + tauyz*sjd(i, j, k, 2) + tauzzd*sj(i, j, k, 3) +&
&         tauzz*sjd(i, j, k, 3)
        fmz = tauxz*sj(i, j, k, 1) + tauyz*sj(i, j, k, 2) + tauzz*sj(i, &
&         j, k, 3)
        frhoed = (ubard*tauxx+ubar*tauxxd+vbard*tauxy+vbar*tauxyd+wbard*&
&         tauxz+wbar*tauxzd)*sj(i, j, k, 1) + (ubar*tauxx+vbar*tauxy+&
&         wbar*tauxz)*sjd(i, j, k, 1) + (ubard*tauxy+ubar*tauxyd+vbard*&
&         tauyy+vbar*tauyyd+wbard*tauyz+wbar*tauyzd)*sj(i, j, k, 2) + (&
&         ubar*tauxy+vbar*tauyy+wbar*tauyz)*sjd(i, j, k, 2) + (ubard*&
&         tauxz+ubar*tauxzd+vbard*tauyz+vbar*tauyzd+wbard*tauzz+wbar*&
&         tauzzd)*sj(i, j, k, 3) + (ubar*tauxz+vbar*tauyz+wbar*tauzz)*&
&         sjd(i, j, k, 3) - q_xd*sj(i, j, k, 1) - q_x*sjd(i, j, k, 1) - &
&         q_yd*sj(i, j, k, 2) - q_y*sjd(i, j, k, 2) - q_zd*sj(i, j, k, 3&
&         ) - q_z*sjd(i, j, k, 3)
        frhoe = (ubar*tauxx+vbar*tauxy+wbar*tauxz)*sj(i, j, k, 1) + (&
&         ubar*tauxy+vbar*tauyy+wbar*tauyz)*sj(i, j, k, 2) + (ubar*tauxz&
&         +vbar*tauyz+wbar*tauzz)*sj(i, j, k, 3) - q_x*sj(i, j, k, 1) - &
&         q_y*sj(i, j, k, 2) - q_z*sj(i, j, k, 3)
! update the residuals of cell j and j+1.
        fwd(i, j, k, imx) = fwd(i, j, k, imx) - fmxd
        fw(i, j, k, imx) = fw(i, j, k, imx) - fmx
        fwd(i, j, k, imy) = fwd(i, j, k, imy) - fmyd
        fw(i, j, k, imy) = fw(i, j, k, imy) - fmy
        fwd(i, j, k, imz) = fwd(i, j, k, imz) - fmzd
        fw(i, j, k, imz) = fw(i, j, k, imz) - fmz
        fwd(i, j, k, irhoe) = fwd(i, j, k, irhoe) - frhoed
        fw(i, j, k, irhoe) = fw(i, j, k, irhoe) - frhoe
        fwd(i, j+1, k, imx) = fwd(i, j+1, k, imx) + fmxd
        fw(i, j+1, k, imx) = fw(i, j+1, k, imx) + fmx
        fwd(i, j+1, k, imy) = fwd(i, j+1, k, imy) + fmyd
        fw(i, j+1, k, imy) = fw(i, j+1, k, imy) + fmy
        fwd(i, j+1, k, imz) = fwd(i, j+1, k, imz) + fmzd
        fw(i, j+1, k, imz) = fw(i, j+1, k, imz) + fmz
        fwd(i, j+1, k, irhoe) = fwd(i, j+1, k, irhoe) + frhoed
        fw(i, j+1, k, irhoe) = fw(i, j+1, k, irhoe) + frhoe
      end do
    end do
  end do
! viscous fluxes in the k-direction
  do k=1,kl
    do j=2,jl
      do i=2,il
! compute the vector from the center of cell k to cell k+1           
        ssxd = eighth*(xd(i-1, j-1, k+1, 1)-xd(i-1, j-1, k-1, 1)+xd(i-1&
&         , j, k+1, 1)-xd(i-1, j, k-1, 1)+xd(i, j-1, k+1, 1)-xd(i, j-1, &
&         k-1, 1)+xd(i, j, k+1, 1)-xd(i, j, k-1, 1))
        ssx = eighth*(x(i-1, j-1, k+1, 1)-x(i-1, j-1, k-1, 1)+x(i-1, j, &
&         k+1, 1)-x(i-1, j, k-1, 1)+x(i, j-1, k+1, 1)-x(i, j-1, k-1, 1)+&
&         x(i, j, k+1, 1)-x(i, j, k-1, 1))
        ssyd = eighth*(xd(i-1, j-1, k+1, 2)-xd(i-1, j-1, k-1, 2)+xd(i-1&
&         , j, k+1, 2)-xd(i-1, j, k-1, 2)+xd(i, j-1, k+1, 2)-xd(i, j-1, &
&         k-1, 2)+xd(i, j, k+1, 2)-xd(i, j, k-1, 2))
        ssy = eighth*(x(i-1, j-1, k+1, 2)-x(i-1, j-1, k-1, 2)+x(i-1, j, &
&         k+1, 2)-x(i-1, j, k-1, 2)+x(i, j-1, k+1, 2)-x(i, j-1, k-1, 2)+&
&         x(i, j, k+1, 2)-x(i, j, k-1, 2))
        sszd = eighth*(xd(i-1, j-1, k+1, 3)-xd(i-1, j-1, k-1, 3)+xd(i-1&
&         , j, k+1, 3)-xd(i-1, j, k-1, 3)+xd(i, j-1, k+1, 3)-xd(i, j-1, &
&         k-1, 3)+xd(i, j, k+1, 3)-xd(i, j, k-1, 3))
        ssz = eighth*(x(i-1, j-1, k+1, 3)-x(i-1, j-1, k-1, 3)+x(i-1, j, &
&         k+1, 3)-x(i-1, j, k-1, 3)+x(i, j-1, k+1, 3)-x(i, j-1, k-1, 3)+&
&         x(i, j, k+1, 3)-x(i, j, k-1, 3))
! and determine one/ length of vector squared
        ssd = -(one*(ssxd*ssx+ssx*ssxd+ssyd*ssy+ssy*ssyd+sszd*ssz+ssz*&
&         sszd)/(ssx*ssx+ssy*ssy+ssz*ssz)**2)
        ss = one/(ssx*ssx+ssy*ssy+ssz*ssz)
        ssxd = ssd*ssx + ss*ssxd
        ssx = ss*ssx
        ssyd = ssd*ssy + ss*ssyd
        ssy = ss*ssy
        sszd = ssd*ssz + ss*sszd
        ssz = ss*ssz
! now compute each gradient
        ddd = wd(i, j, k+1, ivx) - wd(i, j, k, ivx)
        dd = w(i, j, k+1, ivx) - w(i, j, k, ivx)
        u_xd = ddd*ssx + dd*ssxd
        u_x = dd*ssx
        u_yd = ddd*ssy + dd*ssyd
        u_y = dd*ssy
        u_zd = ddd*ssz + dd*sszd
        u_z = dd*ssz
        ddd = wd(i, j, k+1, ivy) - wd(i, j, k, ivy)
        dd = w(i, j, k+1, ivy) - w(i, j, k, ivy)
        v_xd = ddd*ssx + dd*ssxd
        v_x = dd*ssx
        v_yd = ddd*ssy + dd*ssyd
        v_y = dd*ssy
        v_zd = ddd*ssz + dd*sszd
        v_z = dd*ssz
        ddd = wd(i, j, k+1, ivz) - wd(i, j, k, ivz)
        dd = w(i, j, k+1, ivz) - w(i, j, k, ivz)
        w_xd = ddd*ssx + dd*ssxd
        w_x = dd*ssx
        w_yd = ddd*ssy + dd*ssyd
        w_y = dd*ssy
        w_zd = ddd*ssz + dd*sszd
        w_z = dd*ssz
        ddd = aad(i, j, k+1) - aad(i, j, k)
        dd = aa(i, j, k+1) - aa(i, j, k)
        q_xd = -(ddd*ssx+dd*ssxd)
        q_x = -(dd*ssx)
        q_yd = -(ddd*ssy+dd*ssyd)
        q_y = -(dd*ssy)
        q_zd = -(ddd*ssz+dd*sszd)
        q_z = -(dd*ssz)
        por = half*rfilv
        if (pork(i, j, k) .eq. noflux) por = zero
! compute the laminar and (if present) the eddy viscosities
! multiplied by the porosity. compute the factor in front of
! the gradients of the speed of sound squared for the heat
! flux.
        muld = por*(rlvd(i, j, k)+rlvd(i, j, k+1))
        mul = por*(rlv(i, j, k)+rlv(i, j, k+1))
        if (eddymodel) then
          mued = por*(revd(i, j, k)+revd(i, j, k+1))
          mue = por*(rev(i, j, k)+rev(i, j, k+1))
        end if
        mutd = muld + mued
        mut = mul + mue
        gm1 = half*(gamma(i, j, k)+gamma(i, j, k+1)) - one
        factlamheat = one/(prandtl*gm1)
        factturbheat = one/(prandtlturb*gm1)
        heatcoefd = factlamheat*muld + factturbheat*mued
        heatcoef = mul*factlamheat + mue*factturbheat
! compute the stress tensor and the heat flux vector.
        fracdivd = twothird*(u_xd+v_yd+w_zd)
        fracdiv = twothird*(u_x+v_y+w_z)
        tauxxd = mutd*(two*u_x-fracdiv) + mut*(two*u_xd-fracdivd)
        tauxx = mut*(two*u_x-fracdiv)
        tauyyd = mutd*(two*v_y-fracdiv) + mut*(two*v_yd-fracdivd)
        tauyy = mut*(two*v_y-fracdiv)
        tauzzd = mutd*(two*w_z-fracdiv) + mut*(two*w_zd-fracdivd)
        tauzz = mut*(two*w_z-fracdiv)
        tauxyd = mutd*(u_y+v_x) + mut*(u_yd+v_xd)
        tauxy = mut*(u_y+v_x)
        tauxzd = mutd*(u_z+w_x) + mut*(u_zd+w_xd)
        tauxz = mut*(u_z+w_x)
        tauyzd = mutd*(v_z+w_y) + mut*(v_zd+w_yd)
        tauyz = mut*(v_z+w_y)
        q_xd = heatcoefd*q_x + heatcoef*q_xd
        q_x = heatcoef*q_x
        q_yd = heatcoefd*q_y + heatcoef*q_yd
        q_y = heatcoef*q_y
        q_zd = heatcoefd*q_z + heatcoef*q_zd
        q_z = heatcoef*q_z
! compute the average velocities for the face. remember that
! the velocities are stored and not the momentum.
        ubard = half*(wd(i, j, k, ivx)+wd(i, j, k+1, ivx))
        ubar = half*(w(i, j, k, ivx)+w(i, j, k+1, ivx))
        vbard = half*(wd(i, j, k, ivy)+wd(i, j, k+1, ivy))
        vbar = half*(w(i, j, k, ivy)+w(i, j, k+1, ivy))
        wbard = half*(wd(i, j, k, ivz)+wd(i, j, k+1, ivz))
        wbar = half*(w(i, j, k, ivz)+w(i, j, k+1, ivz))
! compute the viscous fluxes for this j-face.
        fmxd = tauxxd*sk(i, j, k, 1) + tauxx*skd(i, j, k, 1) + tauxyd*sk&
&         (i, j, k, 2) + tauxy*skd(i, j, k, 2) + tauxzd*sk(i, j, k, 3) +&
&         tauxz*skd(i, j, k, 3)
        fmx = tauxx*sk(i, j, k, 1) + tauxy*sk(i, j, k, 2) + tauxz*sk(i, &
&         j, k, 3)
        fmyd = tauxyd*sk(i, j, k, 1) + tauxy*skd(i, j, k, 1) + tauyyd*sk&
&         (i, j, k, 2) + tauyy*skd(i, j, k, 2) + tauyzd*sk(i, j, k, 3) +&
&         tauyz*skd(i, j, k, 3)
        fmy = tauxy*sk(i, j, k, 1) + tauyy*sk(i, j, k, 2) + tauyz*sk(i, &
&         j, k, 3)
        fmzd = tauxzd*sk(i, j, k, 1) + tauxz*skd(i, j, k, 1) + tauyzd*sk&
&         (i, j, k, 2) + tauyz*skd(i, j, k, 2) + tauzzd*sk(i, j, k, 3) +&
&         tauzz*skd(i, j, k, 3)
        fmz = tauxz*sk(i, j, k, 1) + tauyz*sk(i, j, k, 2) + tauzz*sk(i, &
&         j, k, 3)
        frhoed = (ubard*tauxx+ubar*tauxxd+vbard*tauxy+vbar*tauxyd+wbard*&
&         tauxz+wbar*tauxzd)*sk(i, j, k, 1) + (ubar*tauxx+vbar*tauxy+&
&         wbar*tauxz)*skd(i, j, k, 1) + (ubard*tauxy+ubar*tauxyd+vbard*&
&         tauyy+vbar*tauyyd+wbard*tauyz+wbar*tauyzd)*sk(i, j, k, 2) + (&
&         ubar*tauxy+vbar*tauyy+wbar*tauyz)*skd(i, j, k, 2) + (ubard*&
&         tauxz+ubar*tauxzd+vbard*tauyz+vbar*tauyzd+wbard*tauzz+wbar*&
&         tauzzd)*sk(i, j, k, 3) + (ubar*tauxz+vbar*tauyz+wbar*tauzz)*&
&         skd(i, j, k, 3) - q_xd*sk(i, j, k, 1) - q_x*skd(i, j, k, 1) - &
&         q_yd*sk(i, j, k, 2) - q_y*skd(i, j, k, 2) - q_zd*sk(i, j, k, 3&
&         ) - q_z*skd(i, j, k, 3)
        frhoe = (ubar*tauxx+vbar*tauxy+wbar*tauxz)*sk(i, j, k, 1) + (&
&         ubar*tauxy+vbar*tauyy+wbar*tauyz)*sk(i, j, k, 2) + (ubar*tauxz&
&         +vbar*tauyz+wbar*tauzz)*sk(i, j, k, 3) - q_x*sk(i, j, k, 1) - &
&         q_y*sk(i, j, k, 2) - q_z*sk(i, j, k, 3)
! update the residuals of cell j and j+1.
        fwd(i, j, k, imx) = fwd(i, j, k, imx) - fmxd
        fw(i, j, k, imx) = fw(i, j, k, imx) - fmx
        fwd(i, j, k, imy) = fwd(i, j, k, imy) - fmyd
        fw(i, j, k, imy) = fw(i, j, k, imy) - fmy
        fwd(i, j, k, imz) = fwd(i, j, k, imz) - fmzd
        fw(i, j, k, imz) = fw(i, j, k, imz) - fmz
        fwd(i, j, k, irhoe) = fwd(i, j, k, irhoe) - frhoed
        fw(i, j, k, irhoe) = fw(i, j, k, irhoe) - frhoe
        fwd(i, j, k+1, imx) = fwd(i, j, k+1, imx) + fmxd
        fw(i, j, k+1, imx) = fw(i, j, k+1, imx) + fmx
        fwd(i, j, k+1, imy) = fwd(i, j, k+1, imy) + fmyd
        fw(i, j, k+1, imy) = fw(i, j, k+1, imy) + fmy
        fwd(i, j, k+1, imz) = fwd(i, j, k+1, imz) + fmzd
        fw(i, j, k+1, imz) = fw(i, j, k+1, imz) + fmz
        fwd(i, j, k+1, irhoe) = fwd(i, j, k+1, irhoe) + frhoed
        fw(i, j, k+1, irhoe) = fw(i, j, k+1, irhoe) + frhoe
      end do
    end do
  end do
end subroutine viscousfluxapprox_d