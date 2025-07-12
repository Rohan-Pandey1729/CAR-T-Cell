%% set IO dirs 
genedparam_dir = "../GenedParams/";
dataset_dir = "../DataSets/";



%% load generated param file
genparam_set_file = '../Batch7.csv';
fi = fopen(genparam_set_file);
    C = textscan(fi, '%s %f %f %f %f %f %f %f %f %f %f %f %f','headerlines',1,'delimiter',',');
fclose(fi);

P = cell(length(C{1}),1);

fi = fopen(genparam_set_file);
    
    % Skip the header
    foffset = 1;
    for koff = 1:foffset
        fgetl(fi);
    end
    
    k1 = 1; 
    while ~feof(fi)
        P{k1} = fgetl(fi);
        k1 = k1+1;
    end
fclose(fi);

for kgen = 1:length(C{1})
        
        
        %% generate output filename
        dataset_filename = "gened_virtual_" + C{1}{kgen} + ".csv";
        genedparam_filename = "gened_virtual_"  + C{1}{kgen} + "_genparam.csv";
        
        %% load params
        Param = [];
        for k1 = 1:10
            Param(k1) = C{k1+1}(kgen);
        end
        
        nB0 = C{12}(kgen);
        nTN0 = C{13}(kgen);

        %% run simulation

        %     f0=[1874.144179,0,7.892191922]; % Initial [2200.24,0,16.46]
        %     Param=[0.070779259	1.686986892	0.109101224	3.03E-05	2808.586835	26.14056824	7932.097398	813.2196461	2118.538966	0.395195924];
            f0 = [nB0, 0, nTN0];
            

            rBp=Param(1);
            rTA0=Param(2);
            lTA0=Param(3);
            lTN=Param(4); 
            nMB=Param(5);
            eBp=Param(6);
            KBp=Param(7);
            KBpr=Param(8);
            KBpTN=Param(9);
            ka=Param(10);


            [t,f]=ode45(@R,0:7,f0,[], rBp, rTA0, lTA0, lTN, nMB, eBp, ka, KBp, KBpr, KBpTN);
            
            %% 
            %% write to dataset
            dataset_file = dataset_dir + dataset_filename;
            fi = fopen(dataset_file,'w');
                fprintf(fi, 'pat_id, time, y, y_id, regressor\n');
                for kt = 1:length(t)
                    fprintf(fi, '%s, %f, %f, 1, %f\n', C{1}{kgen}, t(kt), f(kt,2), nB0);
                end
                fprintf(fi, '%s, 0, %f, 2, %f', C{1}{kgen}, nB0, nB0);
            fclose(fi);
            
            %% write to generated param file
            genedparam_file = genedparam_dir + genedparam_filename;
            fi = fopen(genedparam_file, 'w');
                fprintf(fi,'%s\n',P{kgen});
            fclose(fi);
end


