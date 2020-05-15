import macro from 'vtk.js/Sources/macro';
import HttpDataAccessHelper from 'vtk.js/Sources/IO/Core/DataAccessHelper/HttpDataAccessHelper';
import vtkActor from 'vtk.js/Sources/Rendering/Core/Actor';
import vtkDataArray from 'vtk.js/Sources/Common/Core/DataArray';
import vtkColorMaps from 'vtk.js/Sources/Rendering/Core/ColorTransferFunction/ColorMaps';
import vtkColorTransferFunction from 'vtk.js/Sources/Rendering/Core/ColorTransferFunction';
import vtkFullScreenRenderWindow from 'vtk.js/Sources/Rendering/Misc/FullScreenRenderWindow';
import vtkMapper from 'vtk.js/Sources/Rendering/Core/Mapper';
import vtkURLExtract from 'vtk.js/Sources/Common/Core/URLExtract';
import vtkXMLPolyDataReader from 'vtk.js/Sources/IO/XML/XMLPolyDataReader';
import vtkFPSMonitor from 'vtk.js/Sources/Interaction/UI/FPSMonitor';

import {ColorMode, ScalarMode,} from 'vtk.js/Sources/Rendering/Core/Mapper/Constants';

import style from './GeometryViewer.module.css';
import icon from 'favicon-96x96.png';


export function build_brain_surf_window(x, y) {

    let autoInit = true;
    let background = [0, 0, 0];
    let renderWindow;
    let renderer;

    global.pipeline = {};

// Process arguments from URL
    const userParams = vtkURLExtract.extractURLParameters();
    userParams.fileURL = x;

// Background handling
    if (userParams.background) {
        background = userParams.background.split(',').map((s) => Number(s));
    }
    const selectorClass =
        background.length === 3 && background.reduce((a, b) => a + b, 0) < 1.5
            ? style.dark
            : style.light;

// lut
    const lutName = userParams.lut || 'erdc_rainbow_bright';

// field
    const field = userParams.field || '';

// camera
    function updateCamera(camera) {
        ['zoom', 'pitch', 'elevation', 'yaw', 'azimuth', 'roll', 'dolly'].forEach(
            (key) => {
                if (userParams[key]) {
                    camera[key](userParams[key]);
                }
                renderWindow.render();
            }
        );
    }

// ----------------------------------------------------------------------------
// DOM containers for UI control
// ----------------------------------------------------------------------------

    const rootControllerContainer = document.createElement('div');
    rootControllerContainer.setAttribute('class', style.rootController);

    const addDataSetButton = document.createElement('img');
    addDataSetButton.setAttribute('class', style.button);
    addDataSetButton.setAttribute('src', icon);
    addDataSetButton.addEventListener('click', () => {
        const isVisible = rootControllerContainer.style.display !== 'none';
        rootControllerContainer.style.display = isVisible ? 'none' : 'flex';
    });

    const fpsMonitor = vtkFPSMonitor.newInstance();
    const fpsElm = fpsMonitor.getFpsMonitorContainer();
    fpsElm.classList.add(style.fpsMonitor);

// ----------------------------------------------------------------------------
// Add class to body if iOS device
// ----------------------------------------------------------------------------

    const iOS = /iPad|iPhone|iPod/.test(window.navigator.platform);

    if (iOS) {
        document.querySelector('.GeometryViewer-module-fullScreen_38Xg_').classList.add('is-ios-device');
    }

// ----------------------------------------------------------------------------

    function emptyContainer(container) {
        fpsMonitor.setContainer(null);
        while (container.firstChild) {
            container.removeChild(container.firstChild);
        }
    }

// ----------------------------------------------------------------------------

    function createViewer(container) {
        const fullScreenRenderer = vtkFullScreenRenderWindow.newInstance({
            background,
            rootContainer: container,
            containerStyle: {height: '100%', width: '100%', position: 'absolute'},
        });
        renderer = fullScreenRenderer.getRenderer();
        renderWindow = fullScreenRenderer.getRenderWindow();
        renderWindow.getInteractor().setDesiredUpdateRate(15);

        container.appendChild(rootControllerContainer);
        container.appendChild(addDataSetButton);

        if (userParams.fps) {
            if (Array.isArray(userParams.fps)) {
                fpsMonitor.setMonitorVisibility(...userParams.fps);
                if (userParams.fps.length === 4) {
                    fpsMonitor.setOrientation(userParams.fps[3]);
                }
            }
            fpsMonitor.setRenderWindow(renderWindow);
            fpsMonitor.setContainer(container);
            fullScreenRenderer.setResizeCallback(fpsMonitor.update);
        }
    }

// ----------------------------------------------------------------------------

    function createPipeline(fileName, fileContents) {
        // Create UI
        const presetSelector = document.createElement('select');
        presetSelector.setAttribute('class', selectorClass);
        presetSelector.innerHTML = vtkColorMaps.rgbPresetNames
            .map(
                (name) =>
                    `<option value="${name}" ${
                        lutName === name ? 'selected="selected"' : ''
                    }>${name}</option>`
            )
            .join('');

        const representationSelector = document.createElement('select');
        representationSelector.setAttribute('class', selectorClass);
        representationSelector.innerHTML = [
            // 'Hidden',
            'Points',
            'Wireframe',
            'Surface',
            'Surface with Edge',
        ]
            .map(
                (name, idx) =>
                    `<option value="${(idx + 1) === 0 ? 0 : 1}:${(idx + 1) < 4 ? idx + 1 - 1 : 2}:${
                        (idx + 1) === 4 ? 1 : 0
                    }">${name}</option>`
            )
            .join('');

        representationSelector.value = '1:2:0';

        const colorBySelector = document.createElement('select');
        colorBySelector.setAttribute('class', selectorClass);

        const componentSelector = document.createElement('select');
        componentSelector.setAttribute('class', selectorClass);
        componentSelector.style.display = 'none';

        const opacitySelector = document.createElement('input');
        opacitySelector.setAttribute('class', selectorClass);
        opacitySelector.setAttribute('type', 'range');
        opacitySelector.setAttribute('value', '100');
        opacitySelector.setAttribute('max', '100');
        opacitySelector.setAttribute('min', '1');

        const labelSelector = document.createElement('label');
        labelSelector.setAttribute('class', selectorClass);
        labelSelector.innerHTML = fileName;

        const controlContainer = document.createElement('div');
        controlContainer.setAttribute('class', style.control);
        // controlContainer.appendChild(labelSelector);
        controlContainer.appendChild(representationSelector);
        controlContainer.appendChild(presetSelector);
        controlContainer.appendChild(colorBySelector);
        controlContainer.appendChild(componentSelector);
        // controlContainer.appendChild(opacitySelector);
        rootControllerContainer.appendChild(controlContainer);

        // VTK pipeline
        const vtpReader = vtkXMLPolyDataReader.newInstance();
        vtpReader.parseAsArrayBuffer(fileContents);

        const lookupTable = vtkColorTransferFunction.newInstance();
        const source = vtpReader.getOutputData(0);
        const mapper = vtkMapper.newInstance({
            interpolateScalarsBeforeMapping: false,
            useLookupTableScalarRange: true,
            lookupTable,
            scalarVisibility: false,
        });
        const actor = vtkActor.newInstance();
        const scalars = source.getPointData().getScalars();
        const dataRange = [].concat(scalars ? scalars.getRange() : [0, 1]);
        let activeArray = vtkDataArray;

        // --------------------------------------------------------------------
        // Color handling
        // --------------------------------------------------------------------

        function applyPreset() {
            const preset = vtkColorMaps.getPresetByName(presetSelector.value);
            lookupTable.applyColorMap(preset);
            lookupTable.setMappingRange(dataRange[0], dataRange[1]);
            lookupTable.updateRange();
        }

        applyPreset();
        presetSelector.addEventListener('change', applyPreset);

        // --------------------------------------------------------------------
        // Representation handling
        // --------------------------------------------------------------------

        function updateRepresentation(event) {
            const [
                visibility,
                representation,
                edgeVisibility,
            ] = event.target.value.split(':').map(Number);
            actor.getProperty().set({representation, edgeVisibility});
            actor.setVisibility(!!visibility);
            renderWindow.render();
        }

        representationSelector.addEventListener('change', updateRepresentation);

        // --------------------------------------------------------------------
        // Opacity handling
        // --------------------------------------------------------------------

        function updateOpacity(event) {
            const opacity = Number(event.target.value) / 100;
            actor.getProperty().setOpacity(opacity);
            renderWindow.render();
        }

        opacitySelector.addEventListener('input', updateOpacity);

        // --------------------------------------------------------------------
        // ColorBy handling { value: ':', label: 'Solid color' }
        // --------------------------------------------------------------------

        function toTitleCase(str) {
            return str.replace(/\w\S*/g, function (txt) {
                return txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase();
            });
        }

        function cleanLabel(a) {
            //.split("_").pop()
            let tmp = a.getName().toLowerCase();
            if (tmp.includes("drawem") || tmp.includes("segmentation")) {
                tmp = "Labels";
            } else if (tmp.includes("corrected_thickness")) {
                tmp = "Corrected Thickness";
            } else if (tmp.includes("curvature")) {
                tmp = "Curvature";
            } else if (tmp.includes("sulcal_depth")) {
                tmp = "Sulcal Depth";
            } else if (tmp.includes("myelin_mapl")) {
                tmp = "Myelin Map";
            } else {
                tmp = toTitleCase(tmp);
            }
            return tmp;
        }

        const colorByOptions = [].concat(
            source
                .getPointData()
                .getArrays().filter(function (a) {
                let tmp = a.getName().toLowerCase();
                return !(tmp.includes("roi") || tmp.includes("initial_thickness") || tmp.includes("smooth_myelin_map"))
            })
                .map((a) => ({
                    label: `${cleanLabel(a)}`,
                    value: `PointData:${a.getName()}`,
                })),
            source
                .getCellData()
                .getArrays().filter(function (a) {
                let tmp = a.getName().toLowerCase();
                return !(tmp.includes("roi") || tmp.includes("initial_thickness"))
            })
                .map((a) => ({
                    label: `(c) ${a.getName()}`,
                    value: `CellData:${a.getName()}`,
                }))
        );
        colorBySelector.innerHTML = colorByOptions
            .map(
                ({label, value}) =>
                    `<option value="${value}" ${
                        field === value ? 'selected="selected"' : ''
                    }>${label}</option>`
            )
            .join('');

        function updateColorBy(event) {
            const [location, colorByArrayName] = event.target.value.split(':');
            const interpolateScalarsBeforeMapping = location === 'PointData';
            let colorMode = ColorMode.DEFAULT;
            let scalarMode = ScalarMode.DEFAULT;
            const scalarVisibility = location.length > 0;
            if (scalarVisibility) {
                const newArray = source[`get${location}`]().getArrayByName(
                    colorByArrayName
                );
                activeArray = newArray;
                const newDataRange = activeArray.getRange();
                dataRange[0] = newDataRange[0];
                dataRange[1] = newDataRange[1];
                colorMode = ColorMode.MAP_SCALARS;
                scalarMode =
                    location === 'PointData'
                        ? ScalarMode.USE_POINT_FIELD_DATA
                        : ScalarMode.USE_CELL_FIELD_DATA;

                const numberOfComponents = activeArray.getNumberOfComponents();
                if (numberOfComponents > 1) {
                    // always start on magnitude setting
                    if (mapper.getLookupTable()) {
                        const lut = mapper.getLookupTable();
                        lut.setVectorModeToMagnitude();
                    }
                    componentSelector.style.display = 'block';
                    const compOpts = ['Magnitude'];
                    while (compOpts.length <= numberOfComponents) {
                        compOpts.push(`Component ${compOpts.length}`);
                    }
                    componentSelector.innerHTML = compOpts
                        .map((t, index) => `<option value="${index - 1}">${t}</option>`)
                        .join('');
                } else {
                    componentSelector.style.display = 'none';
                }
            } else {
                componentSelector.style.display = 'none';
            }
            mapper.set({
                colorByArrayName,
                colorMode,
                interpolateScalarsBeforeMapping,
                scalarMode,
                scalarVisibility,
            });
            applyPreset();
        }

        colorBySelector.addEventListener('change', updateColorBy);
        updateColorBy({target: colorBySelector});

        function updateColorByComponent(event) {
            if (mapper.getLookupTable()) {
                const lut = mapper.getLookupTable();
                if (event.target.value === -1) {
                    lut.setVectorModeToMagnitude();
                } else {
                    lut.setVectorModeToComponent();
                    lut.setVectorComponent(Number(event.target.value));
                    const newDataRange = activeArray.getRange(Number(event.target.value));
                    dataRange[0] = newDataRange[0];
                    dataRange[1] = newDataRange[1];
                    lookupTable.setMappingRange(dataRange[0], dataRange[1]);
                    lut.updateRange();
                }
                renderWindow.render();
            }
        }

        componentSelector.addEventListener('change', updateColorByComponent);

        // --------------------------------------------------------------------
        // Pipeline handling
        // --------------------------------------------------------------------

        actor.setMapper(mapper);
        mapper.setInputData(source);
        renderer.addActor(actor);

        // Manage update when lookupTable change
        lookupTable.onModified(() => {
            renderWindow.render();
        });

        // First render
        renderer.resetCamera();
        renderWindow.render();

        global.pipeline[fileName] = {
            actor,
            mapper,
            source,
            lookupTable,
            renderer,
            renderWindow,
        };

        // Update stats
        fpsMonitor.update();
    }

// ----------------------------------------------------------------------------

    function loadFile(file) {
        const reader = new FileReader();
        reader.onload = function onLoad(e) {
            createPipeline(file.name, reader.result);
        };
        reader.readAsArrayBuffer(file);
    }

// ----------------------------------------------------------------------------

    function load(container, options) {
        autoInit = false;
        emptyContainer(container);

        if (options.files) {
            createViewer(container);
            let count = options.files.length;
            while (count--) {
                loadFile(options.files[count]);
            }
            updateCamera(renderer.getActiveCamera());
        } else if (options.fileURL) {
            const urls = [].concat(options.fileURL);
            const progressContainer = document.createElement('div');
            progressContainer.setAttribute('class', style.progress);
            container.appendChild(progressContainer);

            const progressCallback = (progressEvent) => {
                if (progressEvent.lengthComputable) {
                    const percent = Math.floor(
                        (100 * progressEvent.loaded) / progressEvent.total
                    );
                    progressContainer.innerHTML = `Loading ${percent}%`;
                } else {
                    progressContainer.innerHTML = macro.formatBytesToProperUnit(
                        progressEvent.loaded
                    );
                }
            };

            createViewer(container);
            const nbURLs = urls.length;
            let nbLoadedData = 0;

            /* eslint-disable no-loop-func */
            while (urls.length) {
                const url = urls.pop();
                const name = Array.isArray(userParams.name)
                    ? userParams.name[urls.length]
                    : `Data ${urls.length + 1}`;
                HttpDataAccessHelper.fetchBinary(url, {
                    progressCallback,
                }).then((binary) => {
                    nbLoadedData++;
                    if (nbLoadedData === nbURLs) {
                        container.removeChild(progressContainer);
                    }
                    createPipeline(name, binary);
                    updateCamera(renderer.getActiveCamera());
                });
            }
        }
    }

// Look at URL an see if we should load a file
// ?fileURL=https://data.kitware.com/api/v1/item/59cdbb588d777f31ac63de08/download
    if (userParams.url || userParams.fileURL) {
        const myContainer = document.querySelector(y);

        if (myContainer) {
            myContainer.classList.add(style.fullScreen);
            //   rootBody.style.margin = '0';
            //   rootBody.style.padding = '0';
        }

        load(myContainer, userParams);
    }

    return 101;
}

