! Hello world program using OpenMP
program main
  use OMP_LIB
  implicit none

  write(*, *) 'About to enter the parallel world...'

!  write(*, *) 'Number of threads = ', OMP_GET_NUM_THREADS()
  
  !$OMP PARALLEL

  write(*, *) 'Hello world from thread number', OMP_GET_THREAD_NUM()

!  write(*, *) 'Number of threads = ', OMP_GET_NUM_THREADS()

  
  !$OMP END PARALLEL 

  write(*, *) 'Entered the serial world...'

  !write(*, *) 'Number of threads = ', OMP_GET_NUM_THREADS() 
    
  
end program main
