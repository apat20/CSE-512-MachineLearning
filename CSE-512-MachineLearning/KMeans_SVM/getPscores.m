% This function is used to compute the p scores for each iteration of
% k means clustering.

function p_scores = getPscores(label, prediction)

    % Getting the p scores: p1, p2 and p3

    m = size(prediction,1);
    p1 = 0; p2 = 0; 
    count_p1 = 0; count_p2 = 0;

    % Loop conditions are such to avoid double counting
    for i = 1:m-1
        for j = i+1:m
            if label(i) == label(j)
                count_p1 = count_p1 + 1;
                if prediction(i) == prediction(j)
                    p1 = p1 + 1;
                end
            else 
                count_p2 = count_p2 + 1;
                if prediction(i) ~= prediction(j)
                    p2 = p2 + 1;
                end
            end
        end
    end

    % Get the percent values for p1 and p2
    p1 = p1/count_p1;
    p2 = p2/count_p2;
    p3 = (p1 + p2)/2;
    p_scores = [p1,p2,p3];

end