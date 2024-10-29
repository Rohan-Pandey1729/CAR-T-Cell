    %% set IO dirs 
dataset_dir = "../DataSets/";
param_dir = "../Outputs/";
png_dir = "../Pngs/";
genedparam_dir = "../GenedParams/";
pred_dir = "../Preds/";

Stat_file = "../statistics.csv";

%% get input file
% D = dir(dataset_dir);
D = dir(param_dir);

%% Bad files to be masked
file_masked = ["gened_virtual_PR85_out.txt", "gened_virtual_R32_out.txt"];

%% Purge the stat data file
fi = fopen(Stat_file,'w');
    fprintf(fi, 'filename, pat_id, category, T_AUC28, T_max, T_max_t, FT, gen_T_AUC28, gen_T_max, gen_T_max_t, gen_FT\n');
fclose(fi);

%% The main loop
for kdir = 1:length(D)
        if isempty(regexp(D(kdir).name, '.*_out\.txt', 'once'))
            continue
        end
        param_filename = D(kdir).name;
%     dataset_filename = D(kdir).name;
            %% Debugging files
%                 param_filename = "gened_virtual_PR85_out.txt";
        %% Mask bad files
        is_bad_file = false;
        
        for k2 = 1:length(file_masked)
            if strcmp(file_masked(k2), param_filename)
                is_bad_file = true;
                break;
            end
        end
        if is_bad_file
            fprintf('Bad file [%s] is masked out', param_filename); 
            continue;
        end
        
        %% generate file names
%         dataset_filename ='R_1_virtual.csv';
        filename_spl = strsplit(param_filename,'_out.txt');
        base_filename = join(filename_spl{1:end-1},'');

        dataset_filename = base_filename+".csv";
        png_filename = base_filename+"_pic.png";
        genedparam_filename = base_filename+"_genparam.csv";
        pred_filename = base_filename + "_pred.csv";
        
        fprintf('%s\n',base_filename);
        %% parse fit param file
        param_file = param_dir+param_filename;
        fi = fopen(param_file);
            param_name_line = fgetl(fi);
            param_value_line = fgetl(fi);
        fclose(fi);
        exparam_names = strsplit(param_name_line,',');
        exparam_value_strs = strsplit(param_value_line,',');

        exparam_value = nan(1,length(exparam_value_strs));
        for k1 = 1:length(exparam_names)
            exparam_names(k1) = strtrim(exparam_names(k1));
            exparam_value(k1) = str2double(exparam_value_strs(k1));
        end

        param_faces = {'rBp','rTA0','lTA0','lTN','nMB','eBp','KBp','KBpr','KBpTN','ka'}; % must be consisten with Param defined below
        param = nan(1,length(param_faces));
        for k3 = 1:length(exparam_names)
            exname = exparam_names(k3);
            for k2 = 1:length(param_faces)
                face = param_faces(k2);
                if strcmp(face,exname)
                    param(k2) = exparam_value(k3);
                end
            end
            if strcmp(exname,'nTN0')
                nTN0 = exparam_value(k3);
            end
        end
        assert(all(isfinite(param)));
        %% parse dataset for init nB and data points
        dataset_file = dataset_dir+dataset_filename;
        fi = fopen(dataset_file);
            C = textscan(fi, '%s %f %f %f %f','headerLines',1,'Delimiter',',');
        fclose(fi);

        nB0 = C{5}(1);
        y_ids = C{4};
        times = C{2};
        y_vals = C{3};
        nTA_pts = [];
        for k1 = 1:length(y_ids)
            if y_ids(k1) == 1
                nTA_pts = [nTA_pts; times(k1) y_vals(k1)];
            end
        end

        %% Parse gened virtual patient
        genedparam_file = genedparam_dir + genedparam_filename;
        fi = fopen(genedparam_file);
            genC = textscan(fi, '%s %f %f %f %f %f %f %f %f %f %f %f %f','headerlines',0,'delimiter',',');
        fclose(fi);
        
        genParam = [];
        for k1 = 1:10
            genParam(k1) = genC{k1+1};
        end
        gen_nB0 = genC{12};
        gen_nTN0 = genC{13};
        
        %% set t span
        tspan = [0:1:300];
        
        %% run simulation
        
        %     f0=[1874.144179,0,7.892191922]; % Initial [2200.24,0,16.46]
        %     Param=[0.070779259	1.686986892	0.109101224	3.03E-05	2808.586835	26.14056824	7932.097398	813.2196461	2118.538966	0.395195924];
        f0 = [nB0, 0, nTN0];
        Param = param;

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
        
        FT = rTA0*eBp*ka/lTA0;

        [t,f]=ode45(@R,tspan,f0,[], rBp, rTA0, lTA0, lTN, nMB, eBp, ka, KBp, KBpr, KBpTN);
%         [t,f]=ode23s(@R,tspan,f0,[], rBp, rTA0, lTA0, lTN, nMB, eBp, ka, KBp, KBpr, KBpTN);

        %% run virtual experiment
        gen_f0 = [gen_nB0, 0, gen_nTN0];
        

        rBp=genParam(1);
        rTA0=genParam(2);
        lTA0=genParam(3);
        lTN=genParam(4); 
        nMB=genParam(5);
        eBp=genParam(6);
        KBp=genParam(7);
        KBpr=genParam(8);
        KBpTN=genParam(9);
        ka=genParam(10);
        
        gen_FT = rTA0*eBp*ka/lTA0;

        [gen_t,gen_f]=ode45(@R,tspan,gen_f0,[], rBp, rTA0, lTA0, lTN, nMB, eBp, ka, KBp, KBpr, KBpTN);
%         [gen_t,gen_f]=ode23s(@R,tspan,gen_f0,[], rBp, rTA0, lTA0, lTN, nMB, eBp, ka, KBp, KBpr, KBpTN);
    


        %% Calculate T_max and T_AUC28
        kt28 = find(t>=28,1);
        [T_max, T_max_kt] = max(f(1:kt28,2));
        T_max_t = t(T_max_kt);
        T_AUC28 = trapz(t(1:kt28), f(1:kt28,2));
        
        
        genkt28 = find(gen_t>=28,1);
        [gen_T_max, gen_T_max_kt] = max(gen_f(1:genkt28,2));
        gen_T_max_t = gen_t(gen_T_max_kt);
        gen_T_AUC28 = trapz(gen_t(1:genkt28), gen_f(1:genkt28,2));
        
        %% plot and save
        png_file = png_dir+png_filename;
        figure();
        
        subplot(2,3,1)
        hold on 
        plot(t,f(:,1));
        plot(gen_t,gen_f(:,1),'--');
        title('nB+');
        

        subplot(2,3,2)
        hold on
        plot(t,f(:,2));
        plot(gen_t,gen_f(:,2),'--');
        scatter(nTA_pts(:,1),nTA_pts(:,2),'rs')
        title('nTA');

        subplot(2,3,3)
        hold on
        plot(t,f(:,3));
        plot(gen_t,gen_f(:,3),'--');
        title('nTN');

        
        X = categorical({'Fitted','Virtual'});
        X = reordercats(X,{'Fitted','Virtual'});
        subplot(2,3,4);
        hold on
        bar(X,[T_AUC28, gen_T_AUC28]);
        title('T\_AUC28');
        
        subplot(2,3,5);
        hold on
        bar(X,[T_max, gen_T_max]);
        title('T\_max');
        
        subplot(2,3,6);
        hold on
        bar(X,[T_max_t, gen_T_max_t]);
        title('T\_max\_t');
        
        saveas(gcf,png_file);

        %nB_percent=f(:,1)/2159.9*100;
        
        %% Write to prism compatible csv file
        pred_file = pred_dir + pred_filename;
        fi = fopen(pred_file,'w');
            for kt = 1:length(tspan)
                fprintf(fi, '%f,%f,%f,,%f\n', tspan(kt), f(kt,2), nb2tb(f(kt,1)), nb2tb(gen_f(kt,1)));
            end
            for kt = 1:length(nTA_pts(:,1))
                fprintf(fi, '%f,,,%f,\n',nTA_pts(kt,1),nTA_pts(kt,2));
            end
        fclose(fi);
        
        %% Write stat data string
        basename_spl = strsplit(base_filename,"gened_virtual_");
        pat_id = basename_spl{2};
        
        categ_match = regexp(pat_id, '[^0-9]*','match');
        categ = categ_match{1};
        
        
        % base_name, pat_id, category, T_AUC28, T_max, T_max_t,FT, gen_T_AUC28,
        % gen_T_max, gen_T_max_t, gen_FT
        stat_str = sprintf('%s,%s,%s,%f,%f,%f,%f,%f,%f,%f,%f\n', ...
                        base_filename, pat_id, categ,...
                        T_AUC28, T_max, T_max_t, FT,...
                        gen_T_AUC28, gen_T_max, gen_T_max_t, gen_FT );
        fi = fopen(Stat_file,'a');
            fprintf(fi,stat_str);
        fclose(fi);
    
end



