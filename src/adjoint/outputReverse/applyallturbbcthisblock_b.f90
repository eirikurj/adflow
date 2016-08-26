!        generated by tapenade     (inria, tropics team)
!  tapenade 3.10 (r5363) -  9 sep 2014 09:53
!
!  differentiation of applyallturbbcthisblock in reverse (adjoint) mode (with options i4 dr8 r8 noisize):
!   gradient     of useful results: *rev *w
!   with respect to varying inputs: *rev *bvtj1 *bvtj2 *w *bvtk1
!                *bvtk2 *bvti1 *bvti2
!   plus diff mem management of: rev:in bvtj1:in bvtj2:in w:in
!                bvtk1:in bvtk2:in bvti1:in bvti2:in bcdata:in
!      ==================================================================
subroutine applyallturbbcthisblock_b(secondhalo)
!
!      ******************************************************************
!      *                                                                *
!      * applyallturbbcthisblock sets the halo values of the            *
!      * turbulent variables and eddy viscosity for the block the       *
!      * variables in blockpointers currently point to.                 *
!      *                                                                *
!      ******************************************************************
!
  use constants
  use blockpointers
  use flowvarrefstate
  use inputphysics
  implicit none
!
!      subroutine arguments.
!
  logical, intent(in) :: secondhalo
!
!      local variables.
!
  integer(kind=inttype) :: nn, i, j, l, m
  real(kind=realtype), dimension(:, :, :, :), pointer :: bmt
  real(kind=realtype), dimension(:, :, :), pointer :: bvt, ww1, ww2
  real(kind=realtype) :: tmp
  real(kind=realtype) :: tmp0
  real(kind=realtype) :: tmp1
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
  real(kind=realtype) :: tmpd1
  real(kind=realtype) :: tmpd0
! loop over the boundary condition subfaces of this block.
bocos:do nn=1,nbocos
! loop over the faces and set the state in
! the turbulent halo cells.
    if (wallfunctions) then
      call pushcontrol3b(0)
    else
      select case  (bcfaceid(nn)) 
      case (imin) 
        ad_from0 = bcdata(nn)%jcbeg
        do j=ad_from0,bcdata(nn)%jcend
          ad_from = bcdata(nn)%icbeg
          i = bcdata(nn)%icend + 1
          call pushinteger4(i - 1)
          call pushinteger4(ad_from)
        end do
        call pushinteger4(j - 1)
        call pushinteger4(ad_from0)
        call pushcontrol3b(2)
      case (imax) 
        ad_from2 = bcdata(nn)%jcbeg
        do j=ad_from2,bcdata(nn)%jcend
          ad_from1 = bcdata(nn)%icbeg
          i = bcdata(nn)%icend + 1
          call pushinteger4(i - 1)
          call pushinteger4(ad_from1)
        end do
        call pushinteger4(j - 1)
        call pushinteger4(ad_from2)
        call pushcontrol3b(3)
      case (jmin) 
        ad_from4 = bcdata(nn)%jcbeg
        do j=ad_from4,bcdata(nn)%jcend
          ad_from3 = bcdata(nn)%icbeg
          i = bcdata(nn)%icend + 1
          call pushinteger4(i - 1)
          call pushinteger4(ad_from3)
        end do
        call pushinteger4(j - 1)
        call pushinteger4(ad_from4)
        call pushcontrol3b(4)
      case (jmax) 
        ad_from6 = bcdata(nn)%jcbeg
        do j=ad_from6,bcdata(nn)%jcend
          ad_from5 = bcdata(nn)%icbeg
          i = bcdata(nn)%icend + 1
          call pushinteger4(i - 1)
          call pushinteger4(ad_from5)
        end do
        call pushinteger4(j - 1)
        call pushinteger4(ad_from6)
        call pushcontrol3b(5)
      case (kmin) 
        ad_from8 = bcdata(nn)%jcbeg
        do j=ad_from8,bcdata(nn)%jcend
          ad_from7 = bcdata(nn)%icbeg
          i = bcdata(nn)%icend + 1
          call pushinteger4(i - 1)
          call pushinteger4(ad_from7)
        end do
        call pushinteger4(j - 1)
        call pushinteger4(ad_from8)
        call pushcontrol3b(6)
      case (kmax) 
        ad_from10 = bcdata(nn)%jcbeg
        do j=ad_from10,bcdata(nn)%jcend
          ad_from9 = bcdata(nn)%icbeg
          i = bcdata(nn)%icend + 1
          call pushinteger4(i - 1)
          call pushinteger4(ad_from9)
        end do
        call pushinteger4(j - 1)
        call pushinteger4(ad_from10)
        call pushcontrol3b(7)
      case default
        call pushcontrol3b(1)
      end select
    end if
! set the value of the eddy viscosity, depending on the type of
! boundary condition. only if the turbulence model is an eddy
! viscosity model of course.
    if (eddymodel) then
      if (bctype(nn) .eq. nswalladiabatic .or. bctype(nn) .eq. &
&         nswallisothermal) then
        call pushcontrol2b(0)
      else
        call pushcontrol2b(1)
      end if
    else
      call pushcontrol2b(2)
    end if
! extrapolate the turbulent variables in case a second halo
! is needed.
    if (secondhalo) then
      call pushcontrol1b(1)
    else
      call pushcontrol1b(0)
    end if
  end do bocos
  bvtj1d = 0.0_8
  bvtj2d = 0.0_8
  bvtk1d = 0.0_8
  bvtk2d = 0.0_8
  bvti1d = 0.0_8
  bvti2d = 0.0_8
  do nn=nbocos,1,-1
    call popcontrol1b(branch)
    if (branch .ne. 0) call turb2ndhalo_b(nn)
    call popcontrol2b(branch)
    if (branch .eq. 0) then
      call bceddywall_b(nn)
    else if (branch .eq. 1) then
      call bceddynowall_b(nn)
    end if
    call popcontrol3b(branch)
    if (branch .lt. 4) then
      if (branch .ge. 2) then
        if (branch .eq. 2) then
          call popinteger4(ad_from0)
          call popinteger4(ad_to0)
          do j=ad_to0,ad_from0,-1
            call popinteger4(ad_from)
            call popinteger4(ad_to)
            do i=ad_to,ad_from,-1
              do l=nt2,nt1,-1
                do m=nt2,nt1,-1
                  wd(2, i, j, m) = wd(2, i, j, m) - bmti1(i, j, l, m)*wd&
&                   (1, i, j, l)
                end do
                bvti1d(i, j, l) = bvti1d(i, j, l) + wd(1, i, j, l)
                wd(1, i, j, l) = 0.0_8
              end do
            end do
          end do
        else
          call popinteger4(ad_from2)
          call popinteger4(ad_to2)
          do j=ad_to2,ad_from2,-1
            call popinteger4(ad_from1)
            call popinteger4(ad_to1)
            do i=ad_to1,ad_from1,-1
              do l=nt2,nt1,-1
                do m=nt2,nt1,-1
                  tmpd = wd(ie, i, j, l)
                  wd(ie, i, j, l) = tmpd
                  wd(il, i, j, m) = wd(il, i, j, m) - bmti2(i, j, l, m)*&
&                   tmpd
                end do
                bvti2d(i, j, l) = bvti2d(i, j, l) + wd(ie, i, j, l)
                wd(ie, i, j, l) = 0.0_8
              end do
            end do
          end do
        end if
      end if
    else if (branch .lt. 6) then
      if (branch .eq. 4) then
        call popinteger4(ad_from4)
        call popinteger4(ad_to4)
        do j=ad_to4,ad_from4,-1
          call popinteger4(ad_from3)
          call popinteger4(ad_to3)
          do i=ad_to3,ad_from3,-1
            do l=nt2,nt1,-1
              do m=nt2,nt1,-1
                wd(i, 2, j, m) = wd(i, 2, j, m) - bmtj1(i, j, l, m)*wd(i&
&                 , 1, j, l)
              end do
              bvtj1d(i, j, l) = bvtj1d(i, j, l) + wd(i, 1, j, l)
              wd(i, 1, j, l) = 0.0_8
            end do
          end do
        end do
      else
        call popinteger4(ad_from6)
        call popinteger4(ad_to6)
        do j=ad_to6,ad_from6,-1
          call popinteger4(ad_from5)
          call popinteger4(ad_to5)
          do i=ad_to5,ad_from5,-1
            do l=nt2,nt1,-1
              do m=nt2,nt1,-1
                tmpd0 = wd(i, je, j, l)
                wd(i, je, j, l) = tmpd0
                wd(i, jl, j, m) = wd(i, jl, j, m) - bmtj2(i, j, l, m)*&
&                 tmpd0
              end do
              bvtj2d(i, j, l) = bvtj2d(i, j, l) + wd(i, je, j, l)
              wd(i, je, j, l) = 0.0_8
            end do
          end do
        end do
      end if
    else if (branch .eq. 6) then
      call popinteger4(ad_from8)
      call popinteger4(ad_to8)
      do j=ad_to8,ad_from8,-1
        call popinteger4(ad_from7)
        call popinteger4(ad_to7)
        do i=ad_to7,ad_from7,-1
          do l=nt2,nt1,-1
            do m=nt2,nt1,-1
              wd(i, j, 2, m) = wd(i, j, 2, m) - bmtk1(i, j, l, m)*wd(i, &
&               j, 1, l)
            end do
            bvtk1d(i, j, l) = bvtk1d(i, j, l) + wd(i, j, 1, l)
            wd(i, j, 1, l) = 0.0_8
          end do
        end do
      end do
    else
      call popinteger4(ad_from10)
      call popinteger4(ad_to10)
      do j=ad_to10,ad_from10,-1
        call popinteger4(ad_from9)
        call popinteger4(ad_to9)
        do i=ad_to9,ad_from9,-1
          do l=nt2,nt1,-1
            do m=nt2,nt1,-1
              tmpd1 = wd(i, j, ke, l)
              wd(i, j, ke, l) = tmpd1
              wd(i, j, kl, m) = wd(i, j, kl, m) - bmtk2(i, j, l, m)*&
&               tmpd1
            end do
            bvtk2d(i, j, l) = bvtk2d(i, j, l) + wd(i, j, ke, l)
            wd(i, j, ke, l) = 0.0_8
          end do
        end do
      end do
    end if
  end do
end subroutine applyallturbbcthisblock_b