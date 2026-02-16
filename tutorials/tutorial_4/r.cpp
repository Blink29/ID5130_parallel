#include <stdio.h>
#include <omp.h>

int main() {
    // 1. Setup the "Construction Crew" of 4 Workers
    omp_set_num_threads(4);
    
    printf("Construction Started with 4 Workers...\n");
    printf("--------------------------------------\n");

    // 2. The Outer Command: "Split up the 4 Walls among the crew!"
    #pragma omp parallel for
    for (int wall = 0; wall < 4; wall++) {
        
        int worker_id = omp_get_thread_num();
        printf("[Outer] Worker %d has claimed WALL #%d\n", worker_id, wall);

        // 3. The Inner Loop: "Lay 3 Bricks for this wall"
        // Note: Even if we wanted to ask for help here, the other workers 
        // are busy with their own walls. This loop runs SEQUENTIALLY.
        #pragma omp parallel for
        for (int brick = 0; brick < 3; brick++) {
            
            // Artificial delay to keep output somewhat orderly
            for(int i=0; i<10000; i++); 

            printf("   -> [Inner] Worker %d is laying BRICK %d for Wall #%d (All alone)\n", 
                   worker_id, brick, wall);
        }
    }

    return 0;
}