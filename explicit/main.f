! gfortran -O2 -ffixed-line-length-none -mcmodel=medium -c keme.f
! gfortran -O2 -ffixed-line-length-none -mcmodel=medium -c pointsource.f
! gfortran -O2 -ffixed-line-length-none -mcmodel=medium main.f keme.o pointsource.o /usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3

      ! parameter(nex=301,ney=301,nez=150,kd=2,nt=10000)
      ! parameter(nex=201,ney=201,nez=100,kd=2,nt=10000)
      ! parameter(nex=721,ney=721,nez=350,kd=1,nt=100000,nobs=3)
      parameter(nex=101,ney=101,nez=50,kd=2,nt=100000,nobs=3)
      parameter(ne=nex*ney*nez,n=(nex+1)*(ney+1)*(nez+1))
      real*8 rm(4,4,n)
      integer cny(8,ne),flag(nex+1,ney+1,nez+1),num(ne)
      real*8 kek(96,96),keg(96,96)
      real*8 c1(kd),c2(kd),rho(kd),ds,dt,Kl(kd),Gl(kd)
      real*8 rmtmp(4,4,8),rhotmp
      real*8 source(nt),rt,strike,dip,rake,moment,tim,pi
      real*8 fai,delta,ramda,coem(3,3),rmt(4,4)
      real*8 uns(4,4),vns(4,4),sns(4),rmtinv(4,4),snsm(4,4)
!
      real*8 up(12*n),un(12*n),um(12*n),ut(96),ft(96),upt(12),fpt(12)
      real*8 up2(12*n),upm(4,3),fpm(4,3)
!
      real*8 obs(2,nobs),fault(3),iobs(nobs),tobs(3,nobs)
!
      ds=270.0
      dt=0.012
      c1(1)=3900
      c2(1)=2250
      rho(1)=2500
      c1(2)=7800
      c2(2)=4500
      rho(2)=3000
!
! 現状は要素中心に設定する
      fault(1)=(nex+1)/2*ds-0.5*ds
      fault(2)=(ney+1)/2*ds-0.5*ds
      fault(3)=nez*ds-2025
!
      do i=1,nobs
         obs(1,i)=fault(1)+4725+2700*(i-1)
         obs(2,i)=fault(2)+4725+2700*(i-1)
      enddo
!
! risetime
      rt=2.0
      strike=30.
      dip=40.
      rake=50.
      moment=1.0d15
! --------------------------
      ifx=int(fault(1)/ds)+1
      ify=int(fault(2)/ds)+1
      ifz=int(fault(3)/ds)+1
!
      source=1.0
      do it=1,nt
         tim=dt*(it-1)
         if(tim.le.rt/2.)then
            source(it)=2.*tim*tim/rt/rt
         endif
         if((tim.gt.rt/2.).and.(tim.le.rt))then
            source(it)=1.-2.*(tim-rt)**2./rt/rt
         endif
      enddo
!
      source=source*moment
!
      pi=4.*atan(1.)
      fai=pi/180.*strike
      delta=pi/180.*dip
      ramda=pi/180.*rake
      call coemoment(delta,ramda,fai,coem)
!
      do i=1,kd
         Gl(i)=rho(i)*c2(i)**2
         Kl(i)=rho(i)*c1(i)**2-4/3.*Gl(i)
      enddo
!
      in=0
      do k=1,nez+1
         do j=1,ney+1
            do i=1,nex+1
               in=in+1
               flag(i,j,k)=in
            enddo
         enddo
      enddo
      ie=0
      do k=1,nez
         do j=1,ney
            do i=1,nex
               ie=ie+1
!
               if(((k-1)*ds+1/2.*ds).gt.nez*ds-2700)then
                  num(ie)=1
               else
                  num(ie)=2
               endif
!
               cny(1,ie)=flag(i,j,k)
               cny(2,ie)=flag(i+1,j,k)
               cny(3,ie)=flag(i+1,j+1,k)
               cny(4,ie)=flag(i,j+1,k)
               cny(5,ie)=flag(i,j,k+1)
               cny(6,ie)=flag(i+1,j,k+1)
               cny(7,ie)=flag(i+1,j+1,k+1)
               cny(8,ie)=flag(i,j+1,k+1)
            enddo
         enddo
      enddo

      write(6,*) 'obs point (x,y,z)'
      do i=1,nobs
         i1=int(obs(1,i)/ds)+1
         i2=int(obs(2,i)/ds)+1
         id=flag(i1,i2,nez+1)
         iobs(i)=id
         tobs(1,i)=(i1-1)*ds
         tobs(2,i)=(i2-1)*ds
         tobs(3,i)=(nez+1-1)*ds
         write(6,*) tobs(1:3,i)
      enddo
      write(6,*) 'fault location (x,y,z)'
      write(6,*) (ifx-1)*ds+0.5*ds,(ify-1)*ds+0.5*ds,(ifz-1)*ds+0.5*ds

      write(6,*) 'relative (x,y,z)'
      do i=1,nobs
         write(6,*) tobs(1,i)-((ifx-1)*ds+0.5*ds),
     -    tobs(2,i)-((ify-1)*ds+0.5*ds),tobs(3,i)-((ifz-1)*ds+0.5*ds)
      enddo

! --------------------------
      rm=0.
      do ie=1,ne
         in=num(ie)
         rhotmp=rho(in)
         call cmp_me(ds,rhotmp,rmtmp)
         do i1=1,8
            id=cny(i1,ie)
            do j=1,4
               do i=1,4
                  rm(i,j,id)=rm(i,j,id)+rmtmp(i,j,i1)
               enddo
            enddo
         enddo
      enddo
! --------------------------
      snsm=0.
      do id=1,n
         do j=1,4
            do i=1,4
               rmt(i,j)=rm(i,j,id)
            enddo
         enddo
         call svd_mgtn(4,4,rmt,uns,vns,sns)
         if(abs(sns(4)/sns(1)).le.1.0e-6)then
            write(6,*) "rm is singular at node", id
            stop
         endif
!
         do i=1,4
            snsm(i,i)=1/sns(i)
         enddo
         rmtinv=matmul(snsm,transpose(uns))
         rmtinv=matmul(transpose(vns),rmtinv)
!
         do j=1,4
            do i=1,4
               rm(i,j,id)=rmtinv(i,j)
            enddo
         enddo
!
      enddo
! --------------------------
      call def_ke(ds,kek,keg)
! --------------------------
      un=0.
      um=0.
      do it=1,nt
!
         open(60,file='./output.dat',status='unknown')
!
         write(6,*) it
! --------------------------
         up=0.
!
         ie=ifx+(ify-1)*nex+(ifz-1)*nex*ney
         call cmp_eff_fault(coem,ds,ft)
         ft=-ft*source(it)
!
         do i1=1,8
            i2=cny(i1,ie)
            do ii=1,12
               up(12*(i2-1)+ii)=up(12*(i2-1)+ii)+ft(12*(i1-1)+ii)
            enddo
         enddo
! --------------------------
         do ie=1,ne
            in=num(ie)
            do i1=1,8
               i2=cny(i1,ie)
               do ii=1,12
                  ut(12*(i1-1)+ii)=un(12*(i2-1)+ii)
               enddo
            enddo
            ft=Kl(in)*matmul(kek,ut)
            ft=ft+Gl(in)*matmul(keg,ut)
            do i1=1,8
               i2=cny(i1,ie)
               do ii=1,12
                  up(12*(i2-1)+ii)=up(12*(i2-1)+ii)+ft(12*(i1-1)+ii)
               enddo
            enddo
         enddo
! --------------------------
         do j=1,ney+1
            do i=1,nex+1
               id=flag(i,j,1)
               do ii=1,12
                  up(12*(id-1)+ii)=0.
               enddo
            enddo
         enddo
         do k=1,nez+1
            do i=1,nex+1
               id=flag(i,1,k)
               do ii=1,12
                  up(12*(id-1)+ii)=0.
               enddo
               id=flag(i,ney+1,k)
               do ii=1,12
                  up(12*(id-1)+ii)=0.
               enddo
            enddo
         enddo
         do k=1,nez+1
            do j=1,ney+1
               id=flag(1,j,k)
               do ii=1,12
                  up(12*(id-1)+ii)=0.
               enddo
               id=flag(nex+1,j,k)
               do ii=1,12
                  up(12*(id-1)+ii)=0.
               enddo
            enddo
         enddo
! --------------------------

         do id=1,n
            do ii=1,12
               up(12*(id-1)+ii)=-up(12*(id-1)+ii)*dt*dt
            enddo
!
            do j=1,4
               do i=1,4
                  rmtinv(i,j)=rm(i,j,id)
               enddo
            enddo
!
            do i=1,4
               do ii=1,3
                  upm(i,ii)=up(12*(id-1)+3*(i-1)+ii)
               enddo
            enddo
            fpm=matmul(rmtinv,upm)
            do i=1,4
               do ii=1,3
                  up(12*(id-1)+3*(i-1)+ii)=fpm(i,ii)
               enddo
            enddo
!
            do ii=1,12
               up(12*(id-1)+ii)=up(12*(id-1)+ii)+
     -                            2*un(12*(id-1)+ii)-um(12*(id-1)+ii)
!
               um(12*(id-1)+ii)=un(12*(id-1)+ii)
               un(12*(id-1)+ii)=up(12*(id-1)+ii)
            enddo
         enddo
!
! --------------------------
         do i=1,nobs
            id=iobs(i)
            write(6,*) up(12*(id-1)+1),up(12*(id-1)+2),up(12*(id-1)+3)
            write(60,*) up(12*(id-1)+1),up(12*(id-1)+2),up(12*(id-1)+3)
         enddo
!
      enddo

!  100 format(15f10.6)

!
      end
! _______________________________________________________________________
      subroutine coemoment(DIP,RAK,STR,coem)
         real*8 coem(3,3),DIP,RAK,STR
!
         coem(1,1)=2*Cos(STR)*Sin(DIP)*
     -     (-(Cos(DIP)*Cos(STR)*Sin(RAK)) + Cos(RAK)*Sin(STR))
         coem(2,2)=-2*Sin(DIP)*Sin(STR)*
     -     (Cos(RAK)*Cos(STR) + Cos(DIP)*Sin(RAK)*Sin(STR))
         coem(3,3)=2*Cos(DIP)*Sin(DIP)*Sin(RAK)
         coem(1,2)=Sin(DIP)*(Cos(RAK)*Cos(2*STR) +
     -       Cos(DIP)*Sin(RAK)*Sin(2*STR))
         coem(1,3)=-(Cos(DIP)**2*Cos(STR)*Sin(RAK)) +
     -     Cos(STR)*Sin(DIP)**2*Sin(RAK) + Cos(DIP)*Cos(RAK)*Sin(STR)
         coem(2,3)=Cos(DIP)*Cos(RAK)*Cos(STR) + Cos(DIP)**2*Sin(RAK)*Sin(STR) -
     -     Sin(DIP)**2*Sin(RAK)*Sin(STR)
         coem(2,1)=coem(1,2)
         coem(3,2)=coem(2,3)
         coem(3,1)=coem(1,3)
      end
! -----------------------------------------------------------------------
      subroutine svd_mgtn(m,n,a,un,v,sn)
         integer m,n
         real*8 a(m,n)
         integer info
         character jobu,jobv
         integer lda,ldu,ldv,lwork
         real*8 sn(n),un(m,n),v(n,n)
         real*8 work(5*n+m)
!
         jobu = 's'
         jobv = 'a'
         lda = m
         ldu = m
         ldv = n
         lwork = 5 * n + m
!
         call dgesvd (jobu,jobv,m,n,a,lda,sn,un,ldu,v,ldv,
     -     work,lwork,info)
!
         if(info.GT.0) then
            write(*,*)'The algorithm computing SVD failed to converge.'
            stop
         endif
!
      end

