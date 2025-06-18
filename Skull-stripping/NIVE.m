function varargout = NIVE(varargin)
% NIVE MATLAB code for NIVE.fig
%      NIVE, by itself, creates a new NIVE or raises the existing
%      singleton*.
%
%      H = NIVE returns the handle to a new NIVE or the handle to
%      the existing singleton*.
%
%      NIVE('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in NIVE.M with the given input arguments.
%
%      NIVE('Property','Value',...) creates a new NIVE or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before NIVE_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to NIVE_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help NIVE

% Last Modified by GUIDE v2.5 26-Apr-2023 07:12:24

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @NIVE_OpeningFcn, ...
    'gui_OutputFcn',  @NIVE_OutputFcn, ...
    'gui_LayoutFcn',  [] , ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before NIVE is made visible.
function NIVE_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to NIVE (see VARARGIN)

% Choose default command line output for NIVE
handles.output = hObject;
axes(handles.axes4);
utp=imread('picture1copy.png');
imshow(utp)
backgroundd=imread('background.png');
axes(handles.axes1);
imshow(backgroundd);
axes(handles.axes2);
imshow(backgroundd);
axes(handles.axes3);
imshow(backgroundd);
handles.haveInput=0;
% load('NIVE.mat');

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes NIVE wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = NIVE_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on slider movement.
function slider1_Callback(hObject, eventdata, handles)
if handles.haveInput==1
    if isfield(handles,'MRIslices')
        axes(handles.axes1);
        a=round(get(handles.slider1, 'value'));
        %     imshow(handles.MRIslices(:,:,:,a));
        imshow(handles.MRIslices(:,:,a));
        axes(handles.axes2);
        imshow(handles.mask(:,:,a));
        axes(handles.axes3);
        imshow(handles.skullstripped(:,:,a));
    else
        warndlg('Please input MRI first!')
    end
else
    warndlg('Please provide MRI input first!');
end
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider


% --- Executes during object creation, after setting all properties.
function slider1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[file,path] = uigetfile;
if (file~=0)
    handles.haveInput=1;
    [~,~,handles.ext]=fileparts(file);
    switch handles.ext
        case {'.jpg','.jpeg','.png','bmp'}
            set(handles.slider1,'Visible','off');
            set(handles.pushbutton3,'Visible','off');
            handles.MRI=imresize(rgb2gray(imread(strcat(path,file))),[256 256]);
            axes(handles.axes1);
            imshow(handles.MRI);
            load('NIVE.mat');
            checking=semanticseg(handles.MRI,SkullNetMegaDeepLabv3Plus);
            handles.mask=checking=='brain';
            handles.skullstripped=handles.MRI.*uint8(handles.mask);
            axes(handles.axes2);
            imshow(handles.mask);
            axes(handles.axes3);
            imshow(handles.skullstripped);
        case '.dcm'
            set(handles.slider1,'Visible','off');
            set(handles.pushbutton3,'Visible','off');
            handles.MRI=imresize(uint8(255*mat2gray(dicomread(strcat(path,file)))),[256 256]);
            axes(handles.axes1);
            imshow(handles.MRI);
            load('NIVE.mat');
            checking=semanticseg(handles.MRI,SkullNetMegaDeepLabv3Plus);
            handles.mask=checking=='brain';
            handles.skullstripped=handles.MRI.*uint8(handles.mask);
            axes(handles.axes2);
            imshow(handles.mask);
            axes(handles.axes3);
            imshow(handles.skullstripped);

        case '.nii'
            set(handles.slider1,'Visible','on');
            set(handles.pushbutton3,'Visible','on');
            img=niftiread(strcat(path,file));
            slices=size(img,3);
            handles.slcs=slices;
            f1=figure;
            aobj=sliceViewer(img);
            set(gcf,'units','normalized','outerposition',[0 0 1 1]);
            for slice =1: slices
                aobj.SliceNumber=slice;
                hAx = getAxesHandle(aobj);
                I = getframe(hAx);
                currentMask=rgb2gray(I.cdata);
                handles.MRIslices(:,:,slice)=imresize(currentMask,[256 256]);
            end
            close(f1);
            axes(handles.axes1);
            imshow(handles.MRIslices(:,:,round((slices)/2)));
            set(handles.slider1,'min',1);
            set(handles.slider1,'max',slices);
            guidata(hObject, handles);
            set(handles.slider1,'value',round((slices)/2));
            f=waitbar(0,'Please wait, skull stripping in progress...');
            load('NIVE.mat');
            for i = 1 : get(handles.slider1,'Max')
                checking=semanticseg(handles.MRIslices(:,:,i),SkullNetMegaDeepLabv3Plus);
                handles.mask(:,:,i)=checking=='brain';
                handles.skullstripped(:,:,i)=handles.MRIslices(:,:,i).*uint8(handles.mask(:,:,i));
                waitbar(i/get(handles.slider1,'Max'),f);
            end
            axes(handles.axes2);
            imshow(handles.mask(:,:,round(get(handles.slider1,'value'))));
            axes(handles.axes3);
            imshow(handles.skullstripped(:,:,round(get(handles.slider1,'value'))));
        otherwise
            warndlg('The system accepts only jpg, png, bmp, Dicom and NIfTI files as input!');
    end
end
guidata(hObject, handles);
% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
if handles.haveInput==1
    switch handles.ext
        case {'.jpg','.jpeg','.png','bmp'}
            imwrite(handles.skullstripped,'slice.jpg');

        case '.dcm'
            imwrite(handles.skullstripped,'slice.jpg');

        case '.nii'
            a=round(get(handles.slider1,'value'));
            imwrite(handles.skullstripped(:,:,a),strcat('slice',num2str(a),'.jpg'));
            % hObject    handle to pushbutton2 (see GCBO)
            % eventdata  reserved - to be defined in a future version of MATLAB
            % handles    structure with handles and user data (see GUIDATA)
            msgbox('Done!');
    end
else
    warndlg('Please provide MRI input first!');
end
% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
if handles.haveInput==1

    for i = 1 : handles.slcs
        imwrite(handles.skullstripped(:,:,i),strcat('slice',num2str(i),'.jpg'));
    end
    msgbox('Done!');

else

    warndlg('Please provide MRI input first!');
end
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
close all
clear all
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
