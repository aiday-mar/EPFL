x_1 = [1 0];
x_2 = [2 3];
y_1 = [3 4];
y_2 = [0 2];

% plot all the points, the solution must be plotted as lines joining the
% points x_i and y_j if the solution is non zero at position (i, j). 
% implement this graph, try with general points
% automatically give random points function 
% can use constant vector of ones if same number of points
% lines should be thicker if solution is "bigger"
% time it takes to compute the solution for certain number of random points

C = [norm(x_1 - y_1) norm(x_1 - y_2) ; norm(x_2 - y_1) norm(x_2 - y_2)];
c_vector = [C(:,1) ; C(:,2)];
M1 = kron([1 1], eye(2));
M2 = kron(eye(2), [1 1]);

A = [M1; M2];

% now all the weights will be equal to one hence we have vector 
b = [1 ; 1 ; 1 ; 1];

% we also require the z solution to be positive 
z = linprog(c_vector, -1 * eye(4), [0 ; 0 ; 0 ; 0], A, b)

% we get z = [0;1;1;0]

% test the function
X_test = [1 2 ;
          0 3];

Y_test = [3 0;
          4 2];

x = compute_optimal_transport(X_test, Y_test, b)
% the function works correctly


X_test2 = [1 2 3 4 5 ;
           0 3 7 8 9 ;
           4 5 0 1 3 ];

Y_test2 = [3 0 7 8 9;
           4 2 8 1 2;
           6 7 5 3 1];

W2 = [ 1 ; 1; 1; 1; 1; 1; 1; 1; 1; 1];

z = compute_optimal_transport(X_test2, Y_test2, W2);

%plot_result(X_test, Y_test, b)
%plot_result(X_test2, Y_test2, W2)

% try with random number generator
% in 2 dimensions
%[X_rand_2dim, Y_rand_2dim, W_rand_2dim] = random_points(4, 5, 2)
%plot_result(X_rand_2dim, Y_rand_2dim, W_rand_2dim)

% in 3 dimensions 
[X_rand_3dim, Y_rand_3dim, W_rand_3dim] = random_points(3, 2, 3)
plot_result(X_rand_3dim, Y_rand_3dim, W_rand_3dim)

function z = compute_optimal_transport(X, Y, W) 
    
    % where here X contains the x vectors as columns, 
    % and Y contains the y vectors as columns
    % the W contains the weights in the right order for the X columns
    % then the Y columns alread in column format 
    
    % number of columns
    number_x_vectors = size(X, 2);
    number_y_vectors = size(Y, 2);
    % assert false if Y does not have the same number of columns
    
    x_vectors = num2cell(X, 1);
    y_vectors = num2cell(Y, 1);
    C = zeros(number_x_vectors, number_y_vectors);
    c_vector = [];
    
    for i=1:number_x_vectors
        for j = 1:number_y_vectors
            C(i,j) = norm(x_vectors{i} - y_vectors{j});
        end
    end 
    
    for k=1:number_y_vectors
        c_vector = [c_vector ; C(:, k)];
    end
    
    M1 = kron(ones(1,number_x_vectors),eye(number_y_vectors));
    M2 = kron(eye(number_x_vectors), ones(1, number_y_vectors));
    
    A = [M1 ; M2];
    
    total_number_vectors = number_x_vectors * number_y_vectors;
    % here this is the vector p which is calculated
    % the i + n(j − 1) element of p is equal to P_{ij}, this latter element
    % where n is the number of data points x
    % describes the amount of mass flowing from bin i to bin j, from xi to
    % yj
    
    z = linprog(c_vector, -1*eye(total_number_vectors), zeros(total_number_vectors, 1), A, W);
end


function plot_result(X,Y,W)
    
    % we find the number of rows in X
    dimension = size(X,1);
    
    % suppose we have 2D data
    if(dimension == 2)
        
        % the below will contain the optimal solution and now we need to plot
        % the graph
        z = compute_optimal_transport(X,Y,W);
        
        % first let's plot X and Y on a graph, we need all X coordinates placed
        % into one array, all y coordinates into second array, for both the X
        % points and the Y points
        
        % consider the X points first, since the points are noted column wise,
        % this means all 1st coordinates are on the first row, and second
        % coordinates on second row 
        % limited to the 2D case
        
        % X points blue
        %prints correctly 1, 2
        X_xcord = X(1,:);
        % prints correctly 0, 3
        X_ycord = X(2,:);
        scatter(X_xcord, X_ycord, 'filled', 'blue');
        
        % Y points red
        hold on
        Y_xcord = Y(1,:);
        Y_ycord = Y(2,:);
        scatter(Y_xcord, Y_ycord, 'filled', 'red');
        
        % we set the axis 
        max_xcord = max(max(X_xcord), max(Y_xcord)) + 1;
        min_xcord = min(min(X_xcord), min(Y_xcord)) - 1;
        max_ycord = max(max(X_ycord), max(Y_ycord)) + 1;
        min_ycord = min(min(X_ycord), min(Y_ycord)) - 1;
        axis([min_xcord max_xcord min_ycord max_ycord]);
        
        % P is enumerated columnwise, meaning you have fixed y vector
        % and you consider different points x from which you can draw a line to
        % the corresponding point y 
        
        %number of Y points, since the Y points listed as columns hence it's
        %the number of columns of Y
        number_yvectors = size(Y,2);
        number_xvectors = size(X,2);
        
        %find max flow here
        max_flow = max(z);
        
        %fixed y vector, you vary the x_vectors, then change the y_vector to
        %the next
        for j = 1:number_yvectors
            for i = 1:number_xvectors
                if(z(i + number_xvectors*(j-1)) ~= 0)
                    % line has to be drawn between i and j of thickness given
                    % by the corresponding z value
                    % note that the third parameter specifies the width of the
                    % line 
                    X_cord = [X(1,i), Y(1,j)];
                    Y_cord = [X(2,i), Y(2,j)];
                    plot(X_cord,Y_cord, 'black', 'LineWidth', z(i + number_xvectors*(j-1))/max_flow * 2);
                    
                    % we add text next to the line with the actual value
                    % written there
                    % value = int2str( z(i + number_xvectors*(j-1)) );
                    % x_coordinate_text = (X(1,i)+X(2,i))/2;
                    % y_coordinate_text = (Y(1,j)+Y(2,j))/2;
                    % text(x_coordinate_text,y_coordinate_text,value);
                end
            end
        end
    end 
    
    % suppose we are in three dimensions
    if(dimension == 3) 
        
        % the below will contain the optimal solution and now we need to plot
        % the graph
        z = compute_optimal_transport(X,Y,W);
        
        % X points blue
        X_xcord = X(1,:);
        X_ycord = X(2,:);
        X_zcord = X(3,:);
        scatter3(X_xcord, X_ycord, X_zcord, 'filled', 'blue');
        
        % Y points red
        hold on
        Y_xcord = Y(1,:);
        Y_ycord = Y(2,:);
        Y_zcord = Y(3,:);
        scatter3(Y_xcord, Y_ycord, Y_zcord, 'filled', 'red');
        
        % we set the axis 
        max_xcord = max(max(X_xcord), max(Y_xcord)) + 1;
        min_xcord = min(min(X_xcord), min(Y_xcord)) - 1;
        max_ycord = max(max(X_ycord), max(Y_ycord)) + 1;
        min_ycord = min(min(X_ycord), min(Y_ycord)) - 1;
        max_zcord = max(max(X_zcord), max(Y_zcord)) + 1;
        min_zcord = min(min(X_zcord), min(Y_zcord)) - 1;
        axis([min_xcord max_xcord min_ycord max_ycord min_zcord max_zcord]);
        
        % P is enumerated columnwise, meaning you have fixed y vector
        % and you consider different points x from which you can draw a line to
        % the corresponding point y 
        
        %number of Y points, since the Y points listed as columns hence it's
        %the number of columns of Y
        number_yvectors = size(Y,2);
        number_xvectors = size(X,2);
        
        %find max flow here
        max_flow = max(z);
        
        %fixed y vector, you vary the x_vectors, then change the y_vector to
        %the next
        for j = 1:number_yvectors
            for i = 1:number_xvectors
                if(z(i + number_xvectors*(j-1)) ~= 0)
                    % line has to be drawn between i and j of thickness given
                    % by the corresponding z value
                    % note that the third parameter specifies the width of the
                    % line 
                    
                    % seems like here we need arrays for X coordinates, Y
                    % and Z
                    X_cord = [X(1,i), Y(1,j)];
                    Y_cord = [X(2,i), Y(2,j)];
                    Z_cord = [X(3,i), Y(3,j)];
                    plot3(X_cord,Y_cord, Z_cord, 'black', 'LineWidth', z(i + number_xvectors*(j-1))/max_flow * 2);
                    
                end
            end
        end
    end
end

% function used to automatically give random points when we have a
% specified number of x vectors, y vectors and we have dimension given
% random weights should also be provided 
function [X,Y,W] = random_points(number_xvectors, number_yvectors, dim)
    
    % random matrix with values between 0 and 10 of dimensions dim x
    % number_xvectors
    X = randi(10,dim,number_xvectors);
    Y = randi(10,dim,number_yvectors);
    
    %now the weights for all the x vectors summed should equal the weights
    %for all the y_vectors summed
    % choose random sum first
    sum = randi(10);
    
    %random weights for x
    rand_xweights = randfixedsumint(1, number_xvectors, sum).';
    rand_yweights = randfixedsumint(1, number_yvectors, sum).';
    
    W = [rand_xweights ; rand_yweights];
end

function R = randfixedsumint(m,n,S);
   % This generates an m by n array R.  Each row will sum to S, and
   % all elements are all non-negative integers.  The probabilities
   % of each possible set of row elements are all equal.
   % RAS - Mar. 4, 2017
   if ceil(m)~=m|ceil(n)~=n|ceil(S)~=S|m<1|n<1|S<0
    error('Improper arguments')
   else
    P = ones(S+1,n);
    for in = n-1:-1:1
     P(:,in) = cumsum(P(:,in+1));
    end
    R = zeros(m,n);
    for im = 1:m
     s = S;
     for in = 1:n
      R(im,in) = sum(P(s+1,in)*rand<=P(1:s,in));
      s = s-R(im,in);
     end
    end
   end
end 
