!        generated by tapenade     (inria, tropics team)
!  tapenade 3.10 (r5363) -  9 sep 2014 09:53
!
!  differentiation of xhalo_block in reverse (adjoint) mode (with options i4 dr8 r8 noisize):
!   gradient     of useful results: *x
!   with respect to varying inputs: *x
!   rw status of diff variables: *x:in-out
!   plus diff mem management of: x:in
!
!      ******************************************************************
!      *                                                                *
!      * file:          xhalo.f90                                       *
!      * author:        edwin van der weide,c.a.(sandy) mader            *
!      * starting date: 02-23-2003                                      *
!      * last modified: 08-12-2009                                      *
!      *                                                                *
!      ******************************************************************
!
subroutine xhalo_block_b()
!
!      ******************************************************************
!      *                                                                *
!      * xhalo determines the coordinates of the nodal halo's.          *
!      * first it sets all halo coordinates by simple extrapolation,    *
!      * then the symmetry planes are treated (also the unit normal of  *
!      * symmetry planes are determined) and finally an exchange is     *
!      * made for the internal halo's.                                  *
!      *                                                                *
!      ******************************************************************
!
  use constants
  use blockpointers
  use communication
  use inputtimespectral
  implicit none
!
!      local variables.
!
  integer(kind=inttype) :: mm, i, j, k
  integer(kind=inttype) :: ibeg, iend, jbeg, jend, iimax, jjmax
  logical :: err
  real(kind=realtype) :: length, dot
  real(kind=realtype) :: dotd
  real(kind=realtype), dimension(3) :: v1, v2, norm
  real(kind=realtype), dimension(3) :: v1d
  intrinsic sqrt
  real(kind=realtype) :: tmp
  real(kind=realtype) :: tmp0
  real(kind=realtype) :: tmp1
  real(kind=realtype) :: tmp2
  real(kind=realtype) :: tmp3
  real(kind=realtype) :: tmp4
  real(kind=realtype) :: tmp5
  real(kind=realtype) :: tmp6
  real(kind=realtype) :: tmp7
  real(kind=realtype) :: tmp8
  real(kind=realtype) :: tmp9
  real(kind=realtype) :: tmp10
  real(kind=realtype) :: tmp11
  real(kind=realtype) :: tmp12
  real(kind=realtype) :: tmp13
  real(kind=realtype) :: tmp14
  real(kind=realtype) :: tmp15
  real(kind=realtype) :: tmp16
  integer :: ad_from
  integer :: ad_to
  integer :: ad_from0
  integer :: ad_to0
  integer :: ad_from1
  integer :: ad_to1
  integer :: ad_from2
  integer :: ad_to2
  integer :: ad_from3
  integer :: ad_to3
  integer :: ad_from4
  integer :: ad_to4
  integer :: ad_from5
  integer :: ad_to5
  integer :: ad_from6
  integer :: ad_to6
  integer :: ad_from7
  integer :: ad_to7
  integer :: ad_from8
  integer :: ad_to8
  integer :: ad_from9
  integer :: ad_to9
  integer :: ad_from10
  integer :: ad_to10
  integer :: branch
  real(kind=realtype) :: tmpd
  real(kind=realtype) :: tmpd16
  real(kind=realtype) :: tmpd15
  real(kind=realtype) :: tmpd14
  real(kind=realtype) :: tmpd13
  real(kind=realtype) :: tmpd12
  real(kind=realtype) :: tmpd11
  real(kind=realtype) :: tmpd10
  real(kind=realtype) :: tempd
  real(kind=realtype) :: tmpd9
  real(kind=realtype) :: tempd4
  real(kind=realtype) :: tmpd8
  real(kind=realtype) :: tempd3
  real(kind=realtype) :: tmpd7
  real(kind=realtype) :: tempd2
  real(kind=realtype) :: tmpd6
  real(kind=realtype) :: tempd1
  real(kind=realtype) :: tmpd5
  real(kind=realtype) :: tempd0
  real(kind=realtype) :: tmpd4
  real(kind=realtype) :: tmpd3
  real(kind=realtype) :: tmpd2
  real(kind=realtype) :: tmpd1
  real(kind=realtype) :: tmpd0
!
!          **************************************************************
!          *                                                            *
!          * mirror the halo coordinates adjacent to the symmetry       *
!          * planes                                                     *
!          *                                                            *
!          **************************************************************
!
! loop over boundary subfaces.
loopbocos:do mm=1,nbocos
! the actual correction of the coordinates only takes
! place for symmetry planes.
    if (bctype(mm) .eq. symm) then
! set some variables, depending on the block face on
! which the subface is located.
      call pushreal8(norm(1))
      norm(1) = bcdata(mm)%symnorm(1)
      call pushreal8(norm(2))
      norm(2) = bcdata(mm)%symnorm(2)
      call pushreal8(norm(3))
      norm(3) = bcdata(mm)%symnorm(3)
      length = sqrt(norm(1)**2 + norm(2)**2 + norm(3)**2)
! compute the unit normal of the subface.
      call pushreal8(norm(1))
      norm(1) = norm(1)/length
      call pushreal8(norm(2))
      norm(2) = norm(2)/length
      call pushreal8(norm(3))
      norm(3) = norm(3)/length
! see xhalo_block for comments for below:
      if (length .gt. eps) then
        select case  (bcfaceid(mm)) 
        case (imin) 
          ibeg = jnbeg(mm)
          iend = jnend(mm)
          iimax = jl
          jbeg = knbeg(mm)
          jend = knend(mm)
          jjmax = kl
          if (ibeg .eq. 1) ibeg = 0
          if (iend .eq. iimax) iend = iimax + 1
          if (jbeg .eq. 1) jbeg = 0
          if (jend .eq. jjmax) jend = jjmax + 1
          ad_from0 = jbeg
          do j=ad_from0,jend
            ad_from = ibeg
            i = iend + 1
            call pushinteger4(i - 1)
            call pushinteger4(ad_from)
          end do
          call pushinteger4(j - 1)
          call pushinteger4(ad_from0)
          call pushcontrol4b(7)
        case (imax) 
          ibeg = jnbeg(mm)
          iend = jnend(mm)
          iimax = jl
          jbeg = knbeg(mm)
          jend = knend(mm)
          jjmax = kl
          if (ibeg .eq. 1) ibeg = 0
          if (iend .eq. iimax) iend = iimax + 1
          if (jbeg .eq. 1) jbeg = 0
          if (jend .eq. jjmax) jend = jjmax + 1
          ad_from2 = jbeg
          do j=ad_from2,jend
            ad_from1 = ibeg
            i = iend + 1
            call pushinteger4(i - 1)
            call pushinteger4(ad_from1)
          end do
          call pushinteger4(j - 1)
          call pushinteger4(ad_from2)
          call pushcontrol4b(6)
        case (jmin) 
          ibeg = inbeg(mm)
          iend = inend(mm)
          iimax = il
          jbeg = knbeg(mm)
          jend = knend(mm)
          jjmax = kl
          if (ibeg .eq. 1) ibeg = 0
          if (iend .eq. iimax) iend = iimax + 1
          if (jbeg .eq. 1) jbeg = 0
          if (jend .eq. jjmax) jend = jjmax + 1
          ad_from4 = jbeg
          do j=ad_from4,jend
            ad_from3 = ibeg
            i = iend + 1
            call pushinteger4(i - 1)
            call pushinteger4(ad_from3)
          end do
          call pushinteger4(j - 1)
          call pushinteger4(ad_from4)
          call pushcontrol4b(5)
        case (jmax) 
          ibeg = inbeg(mm)
          iend = inend(mm)
          iimax = il
          jbeg = knbeg(mm)
          jend = knend(mm)
          jjmax = kl
          if (ibeg .eq. 1) ibeg = 0
          if (iend .eq. iimax) iend = iimax + 1
          if (jbeg .eq. 1) jbeg = 0
          if (jend .eq. jjmax) jend = jjmax + 1
          ad_from6 = jbeg
          do j=ad_from6,jend
            ad_from5 = ibeg
            i = iend + 1
            call pushinteger4(i - 1)
            call pushinteger4(ad_from5)
          end do
          call pushinteger4(j - 1)
          call pushinteger4(ad_from6)
          call pushcontrol4b(4)
        case (kmin) 
          ibeg = inbeg(mm)
          iend = inend(mm)
          iimax = il
          jbeg = jnbeg(mm)
          jend = jnend(mm)
          jjmax = jl
          if (ibeg .eq. 1) ibeg = 0
          if (iend .eq. iimax) iend = iimax + 1
          if (jbeg .eq. 1) jbeg = 0
          if (jend .eq. jjmax) jend = jjmax + 1
          ad_from8 = jbeg
          do j=ad_from8,jend
            ad_from7 = ibeg
            i = iend + 1
            call pushinteger4(i - 1)
            call pushinteger4(ad_from7)
          end do
          call pushinteger4(j - 1)
          call pushinteger4(ad_from8)
          call pushcontrol4b(3)
        case (kmax) 
          ibeg = inbeg(mm)
          iend = inend(mm)
          iimax = il
          jbeg = jnbeg(mm)
          jend = jnend(mm)
          jjmax = jl
          if (ibeg .eq. 1) ibeg = 0
          if (iend .eq. iimax) iend = iimax + 1
          if (jbeg .eq. 1) jbeg = 0
          if (jend .eq. jjmax) jend = jjmax + 1
          ad_from10 = jbeg
          do j=ad_from10,jend
            ad_from9 = ibeg
            i = iend + 1
            call pushinteger4(i - 1)
            call pushinteger4(ad_from9)
          end do
          call pushinteger4(j - 1)
          call pushinteger4(ad_from10)
          call pushcontrol4b(2)
        case default
          call pushcontrol4b(8)
        end select
      else
        call pushcontrol4b(1)
      end if
    else
      call pushcontrol4b(0)
    end if
  end do loopbocos
  v1d = 0.0_8
  do 100 mm=nbocos,1,-1
    call popcontrol4b(branch)
    if (branch .lt. 4) then
      if (branch .lt. 2) then
        if (branch .eq. 0) goto 100
      else if (branch .eq. 2) then
        call popinteger4(ad_from10)
        call popinteger4(ad_to10)
        do j=ad_to10,ad_from10,-1
          call popinteger4(ad_from9)
          call popinteger4(ad_to9)
          do i=ad_to9,ad_from9,-1
            tmpd14 = xd(i, j, ke, 3)
            xd(i, j, ke, 3) = 0.0_8
            xd(i, j, nz, 3) = xd(i, j, nz, 3) + tmpd14
            tmpd15 = xd(i, j, ke, 2)
            xd(i, j, ke, 2) = 0.0_8
            xd(i, j, nz, 2) = xd(i, j, nz, 2) + tmpd15
            tmpd16 = xd(i, j, ke, 1)
            dotd = norm(2)*tmpd15 + norm(1)*tmpd16 + norm(3)*tmpd14
            xd(i, j, ke, 1) = 0.0_8
            xd(i, j, nz, 1) = xd(i, j, nz, 1) + tmpd16
            tempd4 = two*dotd
            v1d(1) = v1d(1) + norm(1)*tempd4
            v1d(2) = v1d(2) + norm(2)*tempd4
            v1d(3) = v1d(3) + norm(3)*tempd4
            xd(i, j, kl, 3) = xd(i, j, kl, 3) + v1d(3)
            xd(i, j, nz, 3) = xd(i, j, nz, 3) - v1d(3)
            v1d(3) = 0.0_8
            xd(i, j, kl, 2) = xd(i, j, kl, 2) + v1d(2)
            xd(i, j, nz, 2) = xd(i, j, nz, 2) - v1d(2)
            v1d(2) = 0.0_8
            xd(i, j, kl, 1) = xd(i, j, kl, 1) + v1d(1)
            xd(i, j, nz, 1) = xd(i, j, nz, 1) - v1d(1)
            v1d(1) = 0.0_8
          end do
        end do
      else
        call popinteger4(ad_from8)
        call popinteger4(ad_to8)
        do j=ad_to8,ad_from8,-1
          call popinteger4(ad_from7)
          call popinteger4(ad_to7)
          do i=ad_to7,ad_from7,-1
            xd(i, j, 2, 3) = xd(i, j, 2, 3) + xd(i, j, 0, 3)
            dotd = norm(3)*xd(i, j, 0, 3)
            xd(i, j, 0, 3) = 0.0_8
            xd(i, j, 2, 2) = xd(i, j, 2, 2) + xd(i, j, 0, 2)
            dotd = dotd + norm(2)*xd(i, j, 0, 2)
            xd(i, j, 0, 2) = 0.0_8
            xd(i, j, 2, 1) = xd(i, j, 2, 1) + xd(i, j, 0, 1)
            dotd = dotd + norm(1)*xd(i, j, 0, 1)
            xd(i, j, 0, 1) = 0.0_8
            tempd3 = two*dotd
            v1d(1) = v1d(1) + norm(1)*tempd3
            v1d(2) = v1d(2) + norm(2)*tempd3
            v1d(3) = v1d(3) + norm(3)*tempd3
            xd(i, j, 1, 3) = xd(i, j, 1, 3) + v1d(3)
            xd(i, j, 2, 3) = xd(i, j, 2, 3) - v1d(3)
            v1d(3) = 0.0_8
            xd(i, j, 1, 2) = xd(i, j, 1, 2) + v1d(2)
            xd(i, j, 2, 2) = xd(i, j, 2, 2) - v1d(2)
            v1d(2) = 0.0_8
            xd(i, j, 1, 1) = xd(i, j, 1, 1) + v1d(1)
            xd(i, j, 2, 1) = xd(i, j, 2, 1) - v1d(1)
            v1d(1) = 0.0_8
          end do
        end do
      end if
    else if (branch .lt. 6) then
      if (branch .eq. 4) then
        call popinteger4(ad_from6)
        call popinteger4(ad_to6)
        do j=ad_to6,ad_from6,-1
          call popinteger4(ad_from5)
          call popinteger4(ad_to5)
          do i=ad_to5,ad_from5,-1
            tmpd11 = xd(i, je, j, 3)
            xd(i, je, j, 3) = 0.0_8
            xd(i, ny, j, 3) = xd(i, ny, j, 3) + tmpd11
            tmpd12 = xd(i, je, j, 2)
            xd(i, je, j, 2) = 0.0_8
            xd(i, ny, j, 2) = xd(i, ny, j, 2) + tmpd12
            tmpd13 = xd(i, je, j, 1)
            dotd = norm(2)*tmpd12 + norm(1)*tmpd13 + norm(3)*tmpd11
            xd(i, je, j, 1) = 0.0_8
            xd(i, ny, j, 1) = xd(i, ny, j, 1) + tmpd13
            tempd2 = two*dotd
            v1d(1) = v1d(1) + norm(1)*tempd2
            v1d(2) = v1d(2) + norm(2)*tempd2
            v1d(3) = v1d(3) + norm(3)*tempd2
            xd(i, jl, j, 3) = xd(i, jl, j, 3) + v1d(3)
            xd(i, ny, j, 3) = xd(i, ny, j, 3) - v1d(3)
            v1d(3) = 0.0_8
            xd(i, jl, j, 2) = xd(i, jl, j, 2) + v1d(2)
            xd(i, ny, j, 2) = xd(i, ny, j, 2) - v1d(2)
            v1d(2) = 0.0_8
            xd(i, jl, j, 1) = xd(i, jl, j, 1) + v1d(1)
            xd(i, ny, j, 1) = xd(i, ny, j, 1) - v1d(1)
            v1d(1) = 0.0_8
          end do
        end do
      else
        call popinteger4(ad_from4)
        call popinteger4(ad_to4)
        do j=ad_to4,ad_from4,-1
          call popinteger4(ad_from3)
          call popinteger4(ad_to3)
          do i=ad_to3,ad_from3,-1
            xd(i, 2, j, 3) = xd(i, 2, j, 3) + xd(i, 0, j, 3)
            dotd = norm(3)*xd(i, 0, j, 3)
            xd(i, 0, j, 3) = 0.0_8
            xd(i, 2, j, 2) = xd(i, 2, j, 2) + xd(i, 0, j, 2)
            dotd = dotd + norm(2)*xd(i, 0, j, 2)
            xd(i, 0, j, 2) = 0.0_8
            xd(i, 2, j, 1) = xd(i, 2, j, 1) + xd(i, 0, j, 1)
            dotd = dotd + norm(1)*xd(i, 0, j, 1)
            xd(i, 0, j, 1) = 0.0_8
            tempd1 = two*dotd
            v1d(1) = v1d(1) + norm(1)*tempd1
            v1d(2) = v1d(2) + norm(2)*tempd1
            v1d(3) = v1d(3) + norm(3)*tempd1
            xd(i, 1, j, 3) = xd(i, 1, j, 3) + v1d(3)
            xd(i, 2, j, 3) = xd(i, 2, j, 3) - v1d(3)
            v1d(3) = 0.0_8
            xd(i, 1, j, 2) = xd(i, 1, j, 2) + v1d(2)
            xd(i, 2, j, 2) = xd(i, 2, j, 2) - v1d(2)
            v1d(2) = 0.0_8
            xd(i, 1, j, 1) = xd(i, 1, j, 1) + v1d(1)
            xd(i, 2, j, 1) = xd(i, 2, j, 1) - v1d(1)
            v1d(1) = 0.0_8
          end do
        end do
      end if
    else if (branch .eq. 6) then
      call popinteger4(ad_from2)
      call popinteger4(ad_to2)
      do j=ad_to2,ad_from2,-1
        call popinteger4(ad_from1)
        call popinteger4(ad_to1)
        do i=ad_to1,ad_from1,-1
          tmpd8 = xd(ie, i, j, 3)
          xd(ie, i, j, 3) = 0.0_8
          xd(nx, i, j, 3) = xd(nx, i, j, 3) + tmpd8
          tmpd9 = xd(ie, i, j, 2)
          xd(ie, i, j, 2) = 0.0_8
          xd(nx, i, j, 2) = xd(nx, i, j, 2) + tmpd9
          tmpd10 = xd(ie, i, j, 1)
          dotd = norm(2)*tmpd9 + norm(1)*tmpd10 + norm(3)*tmpd8
          xd(ie, i, j, 1) = 0.0_8
          xd(nx, i, j, 1) = xd(nx, i, j, 1) + tmpd10
          tempd0 = two*dotd
          v1d(1) = v1d(1) + norm(1)*tempd0
          v1d(2) = v1d(2) + norm(2)*tempd0
          v1d(3) = v1d(3) + norm(3)*tempd0
          xd(il, i, j, 3) = xd(il, i, j, 3) + v1d(3)
          xd(nx, i, j, 3) = xd(nx, i, j, 3) - v1d(3)
          v1d(3) = 0.0_8
          xd(il, i, j, 2) = xd(il, i, j, 2) + v1d(2)
          xd(nx, i, j, 2) = xd(nx, i, j, 2) - v1d(2)
          v1d(2) = 0.0_8
          xd(il, i, j, 1) = xd(il, i, j, 1) + v1d(1)
          xd(nx, i, j, 1) = xd(nx, i, j, 1) - v1d(1)
          v1d(1) = 0.0_8
        end do
      end do
    else if (branch .eq. 7) then
      call popinteger4(ad_from0)
      call popinteger4(ad_to0)
      do j=ad_to0,ad_from0,-1
        call popinteger4(ad_from)
        call popinteger4(ad_to)
        do i=ad_to,ad_from,-1
          xd(2, i, j, 3) = xd(2, i, j, 3) + xd(0, i, j, 3)
          dotd = norm(3)*xd(0, i, j, 3)
          xd(0, i, j, 3) = 0.0_8
          xd(2, i, j, 2) = xd(2, i, j, 2) + xd(0, i, j, 2)
          dotd = dotd + norm(2)*xd(0, i, j, 2)
          xd(0, i, j, 2) = 0.0_8
          xd(2, i, j, 1) = xd(2, i, j, 1) + xd(0, i, j, 1)
          dotd = dotd + norm(1)*xd(0, i, j, 1)
          xd(0, i, j, 1) = 0.0_8
          tempd = two*dotd
          v1d(1) = v1d(1) + norm(1)*tempd
          v1d(2) = v1d(2) + norm(2)*tempd
          v1d(3) = v1d(3) + norm(3)*tempd
          xd(1, i, j, 3) = xd(1, i, j, 3) + v1d(3)
          xd(2, i, j, 3) = xd(2, i, j, 3) - v1d(3)
          v1d(3) = 0.0_8
          xd(1, i, j, 2) = xd(1, i, j, 2) + v1d(2)
          xd(2, i, j, 2) = xd(2, i, j, 2) - v1d(2)
          v1d(2) = 0.0_8
          xd(1, i, j, 1) = xd(1, i, j, 1) + v1d(1)
          xd(2, i, j, 1) = xd(2, i, j, 1) - v1d(1)
          v1d(1) = 0.0_8
        end do
      end do
    end if
    call popreal8(norm(3))
    call popreal8(norm(2))
    call popreal8(norm(1))
    call popreal8(norm(3))
    call popreal8(norm(2))
    call popreal8(norm(1))
 100 continue
  do j=je,0,-1
    do i=ie,0,-1
      tmpd5 = xd(i, j, ke, 3)
      xd(i, j, ke, 3) = 0.0_8
      xd(i, j, kl, 3) = xd(i, j, kl, 3) + two*tmpd5
      xd(i, j, nz, 3) = xd(i, j, nz, 3) - tmpd5
      tmpd6 = xd(i, j, ke, 2)
      xd(i, j, ke, 2) = 0.0_8
      xd(i, j, kl, 2) = xd(i, j, kl, 2) + two*tmpd6
      xd(i, j, nz, 2) = xd(i, j, nz, 2) - tmpd6
      tmpd7 = xd(i, j, ke, 1)
      xd(i, j, ke, 1) = 0.0_8
      xd(i, j, kl, 1) = xd(i, j, kl, 1) + two*tmpd7
      xd(i, j, nz, 1) = xd(i, j, nz, 1) - tmpd7
      xd(i, j, 1, 3) = xd(i, j, 1, 3) + two*xd(i, j, 0, 3)
      xd(i, j, 2, 3) = xd(i, j, 2, 3) - xd(i, j, 0, 3)
      xd(i, j, 0, 3) = 0.0_8
      xd(i, j, 1, 2) = xd(i, j, 1, 2) + two*xd(i, j, 0, 2)
      xd(i, j, 2, 2) = xd(i, j, 2, 2) - xd(i, j, 0, 2)
      xd(i, j, 0, 2) = 0.0_8
      xd(i, j, 1, 1) = xd(i, j, 1, 1) + two*xd(i, j, 0, 1)
      xd(i, j, 2, 1) = xd(i, j, 2, 1) - xd(i, j, 0, 1)
      xd(i, j, 0, 1) = 0.0_8
    end do
  end do
  do k=kl,1,-1
    do i=ie,0,-1
      tmpd2 = xd(i, je, k, 3)
      xd(i, je, k, 3) = 0.0_8
      xd(i, jl, k, 3) = xd(i, jl, k, 3) + two*tmpd2
      xd(i, ny, k, 3) = xd(i, ny, k, 3) - tmpd2
      tmpd3 = xd(i, je, k, 2)
      xd(i, je, k, 2) = 0.0_8
      xd(i, jl, k, 2) = xd(i, jl, k, 2) + two*tmpd3
      xd(i, ny, k, 2) = xd(i, ny, k, 2) - tmpd3
      tmpd4 = xd(i, je, k, 1)
      xd(i, je, k, 1) = 0.0_8
      xd(i, jl, k, 1) = xd(i, jl, k, 1) + two*tmpd4
      xd(i, ny, k, 1) = xd(i, ny, k, 1) - tmpd4
      xd(i, 1, k, 3) = xd(i, 1, k, 3) + two*xd(i, 0, k, 3)
      xd(i, 2, k, 3) = xd(i, 2, k, 3) - xd(i, 0, k, 3)
      xd(i, 0, k, 3) = 0.0_8
      xd(i, 1, k, 2) = xd(i, 1, k, 2) + two*xd(i, 0, k, 2)
      xd(i, 2, k, 2) = xd(i, 2, k, 2) - xd(i, 0, k, 2)
      xd(i, 0, k, 2) = 0.0_8
      xd(i, 1, k, 1) = xd(i, 1, k, 1) + two*xd(i, 0, k, 1)
      xd(i, 2, k, 1) = xd(i, 2, k, 1) - xd(i, 0, k, 1)
      xd(i, 0, k, 1) = 0.0_8
    end do
  end do
  do k=kl,1,-1
    do j=jl,1,-1
      tmpd = xd(ie, j, k, 3)
      xd(ie, j, k, 3) = 0.0_8
      xd(il, j, k, 3) = xd(il, j, k, 3) + two*tmpd
      xd(nx, j, k, 3) = xd(nx, j, k, 3) - tmpd
      tmpd0 = xd(ie, j, k, 2)
      xd(ie, j, k, 2) = 0.0_8
      xd(il, j, k, 2) = xd(il, j, k, 2) + two*tmpd0
      xd(nx, j, k, 2) = xd(nx, j, k, 2) - tmpd0
      tmpd1 = xd(ie, j, k, 1)
      xd(ie, j, k, 1) = 0.0_8
      xd(il, j, k, 1) = xd(il, j, k, 1) + two*tmpd1
      xd(nx, j, k, 1) = xd(nx, j, k, 1) - tmpd1
      xd(1, j, k, 3) = xd(1, j, k, 3) + two*xd(0, j, k, 3)
      xd(2, j, k, 3) = xd(2, j, k, 3) - xd(0, j, k, 3)
      xd(0, j, k, 3) = 0.0_8
      xd(1, j, k, 2) = xd(1, j, k, 2) + two*xd(0, j, k, 2)
      xd(2, j, k, 2) = xd(2, j, k, 2) - xd(0, j, k, 2)
      xd(0, j, k, 2) = 0.0_8
      xd(1, j, k, 1) = xd(1, j, k, 1) + two*xd(0, j, k, 1)
      xd(2, j, k, 1) = xd(2, j, k, 1) - xd(0, j, k, 1)
      xd(0, j, k, 1) = 0.0_8
    end do
  end do
end subroutine xhalo_block_b