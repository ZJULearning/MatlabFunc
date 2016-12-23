function [centers, radii] = SphericalHashing(data, bit)
%Spherical Hashing method by Jae-Pil Heo

    [N, D] = size(data);
    centers = random_center(data, bit);
    [O1, O2, radii, avg, stddev] = compute_statistics(data, centers);
    
    iter = 1;
    while true        
        forces = zeros(bit, D);
        for i = 1:bit - 1
            for j = i + 1:bit
                force = 0.5 * (O2(i, j) - N / 4) / (N / 4) * (centers(i, :) - centers(j, :));
                forces(i, :) = forces(i, :) + force ./ bit;
                forces(j, :) = forces(j, :) - force ./ bit;
            end
        end
        centers = centers + forces;
        
        [O1, O2, radii, avg, stddev] = compute_statistics(data, centers);
		
        if avg <= 0.1 * N / 4 && stddev <= 0.15 * N / 4
            break;
        end
        if iter >= 100
            fprintf('iter exceed 100, avg = %f, stddev = %f\n', avg, stddev);
        end
        
        iter = iter + 1;
    end
    %fprintf('iteration = %d\n', iter);
end

function centers = random_center(data, bit)
    [N, D] = size(data);
    centers = zeros(bit, D);
    for i = 1:bit
        R = randperm(N);
        sample = data(R(1:5), :);
        sample = sum(sample, 1) / 5;
        centers(i, :) = sample(:);
    end
end

function [O1, O2, radii, avg, stddev] = compute_statistics(data, centers) 
    [N, D] = size(data);
    bit = size(centers, 1);
    
    dist = EuDist2(centers,data);
    sort_dist = sort(dist, 2);
    radii = sort_dist(:, N / 2);
    dist = dist <= repmat(radii, 1, N);
    dist = dist * 1.0;

    O1 = sum(dist, 2);
    avg = 0;
    avg2 = 0;
    O2 = dist * dist';
    for i = 1:bit-1
        for j = i + 1:bit
            avg = avg + abs(O2(i, j) - N / 4);
            avg2 = avg2 + O2(i, j);
        end
    end
    
    avg = avg / (bit * (bit - 1) / 2);
    avg2 = avg2 / (bit * (bit - 1) / 2);
    stddev = 0;
    for i = 1:bit - 1
        for j = i + 1:bit
            stddev = stddev + (O2(i, j) - avg2) ^ 2;
        end
    end
    stddev = sqrt(stddev / (bit * (bit - 1) / 2));
end
