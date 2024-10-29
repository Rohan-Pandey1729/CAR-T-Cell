function tb = nb2tb(nb)
% tb in percent, i.e. tb < 100
% nb = 1909 .* tb./(97.19-tb);
tb = 97.19 .* nb./(1909+nb);
end