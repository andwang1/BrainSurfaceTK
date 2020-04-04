import 'vtk.js/Sources/favicon';

import vtkActor from 'vtk.js/Sources/Rendering/Core/Actor';
import vtkRenderWindowWithControlBar from 'vtk.js/Sources/Rendering/Misc/RenderWindowWithControlBar';
import vtkMapper from 'vtk.js/Sources/Rendering/Core/Mapper';
import vtkXMLPolyDataReader from 'vtk.js/Sources/IO/XML/XMLPolyDataReader';


const script_tag = document.getElementById('brain_surf');
const fileName = script_tag.getAttribute("data-fileName");
const BASE_PATH = script_tag.getAttribute("data-filePath");

// Define container size/position
const brain_div = document.getElementById('brain-container');
const rootContainer = document.createElement('div');
// rootContainer.style.position = 'relative';
rootContainer.style.width = '100%';
rootContainer.style.height = '100%';
brain_div.appendChild(rootContainer);


// ----------------------------------------------------------------------------
// Standard rendering code setup
// ----------------------------------------------------------------------------

// Create render window inside container
const renderWindow = vtkRenderWindowWithControlBar.newInstance({
  controlSize: 25,
});
renderWindow.setContainer(rootContainer);

// ----------------------------------------------------------------------------
// Example code
// ----------------------------------------------------------------------------

const reader = vtkXMLPolyDataReader.newInstance();
const scene = [];

function onClick(event) {
  const el = event.target;
  const index = Number(el.dataset.index);
  const actor = scene[index].actor;
  const visibility = actor.getVisibility();

  actor.setVisibility(!visibility);
  if (visibility) {
    el.classList.remove('visible');
  } else {
    el.classList.add('visible');
  }
  render();
}

reader
  .setUrl(`${BASE_PATH}/${fileName}.vtp`)
  .then(() => {
    const size = reader.getNumberOfOutputPorts();
    for (let i = 0; i < size; i++) {
      const polydata = reader.getOutputData(i);
      const name = polydata.get('name').name;
      const mapper = vtkMapper.newInstance();
      const actor = vtkActor.newInstance();

      actor.setMapper(mapper);
      mapper.setInputData(polydata);

      renderWindow.getRenderer().addActor(actor);

      scene.push({ name, polydata, mapper, actor });
    }
    renderWindow.getRenderer().resetCamera();
    renderWindow.getRenderWindow().render();

    // Build control ui
    const htmlBuffer = [
      '<style>.visible { font-weight: bold; } .click { cursor: pointer; min-width: 150px;}</style>',
    ];
    scene.forEach((item, idx) => {
      htmlBuffer.push(
        `<div class="click visible" data-index="${idx}">${item.name}</div>`
      );
    });

    renderWindow.getControlContainer(htmlBuffer.join('\n'));
    const nodes = document.querySelectorAll('.click');
    for (let i = 0; i < nodes.length; i++) {
      const el = nodes[i];
      el.onclick = onClick;
    }
  });


