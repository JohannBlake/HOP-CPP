const container = document.getElementById('deck-container');
const slider = document.getElementById('pointCloudSlider');
const animationSpeedInput = document.getElementById('animationSpeed');
const menuButton = document.getElementById('menu-button');
const toggleMetricButton = document.getElementById('toggle-metric-button');
const togglePointCloudButton = document.getElementById('togglePointCloudButton');
const toggleObservationButton = document.getElementById('toggleObservationButton');
const toggleMapButton = document.getElementById('toggleMapButton');
const menuContainer = document.getElementById('menu-container');
const bubbleSizeInput = document.getElementById('bubbleSize');
const pathColorInput = document.getElementById('pathColor');
const pathSizeInput = document.getElementById('pathSize');
const metricContainer = document.getElementById('metric-container');
const configContainer = document.getElementById('config-container');
const observationContainer = document.getElementById('observation-container');
const observationCanvas = document.getElementById('observation-canvas');
const runsDropdown = document.getElementById('runs-dropdown');
const sweepsDropdown = document.getElementById('sweeps-dropdown');

// State object to hold shared variables
const appState = {
  isPlaying: false,
  playInterval: null,
  metricData: [],
  colorsPerTimestep: [],
  updateLayers: null,
  deckGL: null,
  runConfig: null,
  pointClouds: [], // Add pointClouds to hold point cloud data
  currentViewState: null, // Track current view state for transitions
  showPointCloud: true, // Toggle for point cloud visibility
  showBitmapLayer: true, // Toggle for bitmap layer visibility (for 'jon' type)
  showObservation: false, // Toggle for observation image visibility
  showMap: true, // Toggle for map/terrain layer visibility
  observationImages: [], // Store observation images
  observationMetadata: [], // Store observation image metadata
  pngData: [], // Store PNG data for each step (when z_rad_type is 'jon')
  polygonData: [], // Store polygon data for each step (when z_rad_type is 'jon')
  currentSweepId: null, // Current sweep ID
  currentRunId: null, // Current run ID
  currentBitmapLayer: null, // Cache the current bitmap layer
  lastImagePath: null, // Track the last image path to avoid unnecessary updates
  loadedSteps: new Set(), // Track which steps have been loaded
  availableSteps: [] // Store available steps from metadata
};

// Function to load geo_json data for a specific step
async function loadGeoJsonForStep(sweepId, runId, step) {
  if (appState.loadedSteps.has(step)) {
    return; // Already loaded
  }
  
  try {
    const response = await fetch(`html_data/${sweepId}/${runId}/geo_json_chunks/step_${step}.msgpack`);
    if (!response.ok) {
      console.warn(`Geo JSON data not available for step ${step}`);
      return;
    }
    
    const buffer = await response.arrayBuffer();
    const stepGeoJsonData = msgpack.decode(new Uint8Array(buffer));
    const features = stepGeoJsonData.features;
    
    // Process features for this step
    features.forEach(feature => {
      const category = feature.properties.category;
      const stepIndex = feature.properties.current_step_in_animation;
      
      if (category === 'path_of_episode' && feature.properties.run_id === runId) {
        appState.paths[stepIndex] = feature.geometry.coordinates;
      } else if (category === 'target_area') {
        appState.totalMeasuredAreas[stepIndex] = feature.geometry.coordinates;
      }
    });
    
    appState.loadedSteps.add(step);
    console.log(`Loaded geo JSON data for step ${step}`);
  } catch (error) {
    console.warn(`Failed to load geo JSON data for step ${step}:`, error);
  }
}

// Function to load polygon data for a specific step (for z_rad_type == 'jon')
async function loadPolygonDataForStep(sweepId, runId, step) {
  try {
    const response = await fetch(`html_data/${sweepId}/${runId}/polygon_data_chunks/step_${step}.json`);
    if (!response.ok) {
      console.warn(`Polygon data not available for step ${step}`);
      return null;
    }
    
    const polygonData = await response.json();
    appState.polygonData[step] = polygonData;
    console.log(`Loaded polygon data for step ${step}`);
    return polygonData;
  } catch (error) {
    console.warn(`Failed to load polygon data for step ${step}:`, error);
    return null;
  }
}

// Define handleKeyPress
async function handleKeyPress(event) {
  if (event.key === ' ' || event.code === 'Space') {
    event.preventDefault();
    togglePlayStop();
  } else if ((event.key === '+' || event.key === '=') && !appState.isPlaying) {
    event.preventDefault();
    await nextFrame();
  } else if (event.key === '-' && !appState.isPlaying) {
    event.preventDefault();
    await previousFrame();
  } else if (event.key === 'c' || event.key === 'C') {
    event.preventDefault();
    centerOnCurrentPath();
  }
}

// Add event listener once
document.addEventListener('keydown', handleKeyPress);

// Function to center the map on the current path position
function centerOnCurrentPath() {
  const currentIndex = parseInt(slider.value);
  const pathPoints = appState.paths[currentIndex];
  
  if (pathPoints && pathPoints.length > 0) {
    // Get the last position in the current path
    const lastPosition = pathPoints[pathPoints.length - 1];
    
    // Create a new initialViewState with the desired target position
    const newViewState = {
      longitude: lastPosition[0],
      latitude: lastPosition[1],
      zoom: appState.currentViewState ? appState.currentViewState.zoom : 11.5,
      pitch: appState.currentViewState ? appState.currentViewState.pitch : 0,
      bearing: appState.currentViewState ? appState.currentViewState.bearing : 0,
      transitionDuration: 100,
      transitionInterpolator: new deck.FlyToInterpolator()
    };
    
    // Use setProps with initialViewState to maintain controller functionality
    appState.deckGL.setProps({
      initialViewState: newViewState
    });
  }
}

function togglePlayStop() {
  if (appState.isPlaying) {
    clearInterval(appState.playInterval);
  } else {
    const speed = parseInt(animationSpeedInput.value);
    appState.playInterval = setInterval(async () => {
      let currentIndex = parseInt(slider.value);
      currentIndex = (currentIndex + 1) % appState.colorsPerTimestep.length;
      slider.value = currentIndex;
      await appState.updateLayers(currentIndex);
    }, speed);
  }
  appState.isPlaying = !appState.isPlaying;
}

async function nextFrame() {
  let currentIndex = parseInt(slider.value);
  if (currentIndex < appState.colorsPerTimestep.length - 1) {
    currentIndex += 1;
    slider.value = currentIndex;
    await appState.updateLayers(currentIndex);
  }
}

async function previousFrame() {
  let currentIndex = parseInt(slider.value);
  if (currentIndex > 0) {
    currentIndex -= 1;
    slider.value = currentIndex;
    await appState.updateLayers(currentIndex);
  }
}

menuButton.addEventListener('click', () => {
  menuContainer.style.display = menuContainer.style.display === 'none' ? 'block' : 'none';
});

toggleMetricButton.addEventListener('click', () => {
  metricContainer.style.display = metricContainer.style.display === 'none' ? 'block' : 'none';
});

togglePointCloudButton.addEventListener('click', async () => {
  console.log('Toggle button clicked!');
  console.log('Current runConfig:', appState.runConfig);
  console.log('Current radiation_type:', appState.runConfig ? appState.runConfig.radiation_type : 'undefined');
  
  // Check if we have polygon data or PNG data to determine if this is 'jon' type
  const hasJonData = (appState.polygonData && appState.polygonData.length > 0) || 
                     (appState.pngData && appState.pngData.length > 0);
  const isJonType = (appState.runConfig && appState.runConfig.radiation_type === 'jon') || hasJonData;
  
  console.log('Has PNG data (Jon type indicator):', hasJonData);
  console.log('Is Jon type:', isJonType);
  
  if (isJonType) {
    // For 'jon' type, toggle bitmap layer visibility
    const previousState = appState.showBitmapLayer;
    appState.showBitmapLayer = !appState.showBitmapLayer;
    console.log(`Hidden State button clicked - Jon radiation type: toggling bitmap layer from ${previousState} to ${appState.showBitmapLayer}`);
  } else {
    // For 'bfs' and other types, toggle point cloud visibility
    const previousState = appState.showPointCloud;
    appState.showPointCloud = !appState.showPointCloud;
    console.log(`Hidden State button clicked - BFS/other radiation type: toggling point cloud from ${previousState} to ${appState.showPointCloud}`);
  }
  
  // Re-render current frame with new visibility settings
  const currentIndex = parseInt(slider.value);
  console.log('Calling updateLayers with index:', currentIndex);
  if (appState.updateLayers) {
    await appState.updateLayers(currentIndex);
  } else {
    console.error('updateLayers function not available!');
  }
});

toggleObservationButton.addEventListener('click', () => {
  console.log('Observation button clicked, current state:', appState.showObservation);
  appState.showObservation = !appState.showObservation;
  observationContainer.style.display = appState.showObservation ? 'block' : 'none';
  console.log('New state:', appState.showObservation, 'Container display:', observationContainer.style.display);
  if (appState.showObservation) {
    // Display current observation image
    const currentIndex = parseInt(slider.value);
    console.log('Displaying observation image for index:', currentIndex);
    displayObservationImage(currentIndex);
  }
});

toggleMapButton.addEventListener('click', async () => {
  console.log('Map button clicked, current state:', appState.showMap);
  appState.showMap = !appState.showMap;
  console.log('New state:', appState.showMap);
  
  // Re-render current frame with new map visibility settings
  const currentIndex = parseInt(slider.value);
  console.log('Calling updateLayers with index:', currentIndex);
  if (appState.updateLayers) {
    await appState.updateLayers(currentIndex);
  } else {
    console.error('updateLayers function not available!');
  }
});

// Add slider input event listener for manual slider changes
slider.addEventListener('input', async () => {
  if (!appState.isPlaying && appState.updateLayers) {
    const currentIndex = parseInt(slider.value);
    await appState.updateLayers(currentIndex);
  }
});

// Initialize the application by loading sweep IDs
function initializeApp() {
  loadSweepIds();
}

// Call initializeApp to start the process
initializeApp();

function loadSweepIds() {
  fetch('html_data/sweep_ids.json')
    .then(response => {
      if (response.ok) return response.text();
      else throw new Error('Network response was not ok.');
    })
    .then(text => {
      const sweepIds = JSON.parse(text);
      sweepIds.forEach(sweepId => {
        const option = document.createElement('option');
        option.value = sweepId;
        option.text = sweepId;
        sweepsDropdown.appendChild(option);
      });

      // Load run IDs for the first sweep_id by default
      loadRunIds(sweepIds[0]);
    })
    .catch(error => {
      console.error('Error loading sweep_ids:', error);
    });
}

function loadRunIds(sweepId) {
  // Clear the runs-dropdown
  runsDropdown.innerHTML = '';

  fetch(`html_data/${sweepId}/run_ids.json`)
    .then(response => {
      if (response.ok) return response.text();
      else throw new Error('Network response was not ok.');
    })
    .then(text => {
      const runIds = JSON.parse(text);
      runIds.forEach(runId => {
        const option = document.createElement('option');
        option.value = runId;
        option.text = runId;
        runsDropdown.appendChild(option);
      });

      // Load data for the first run_id by default
      loadData(sweepId, runIds[0]);
    })
    .catch(error => {
      console.error('Error loading run_ids:', error);
    });
}

// Add event listeners
sweepsDropdown.addEventListener('change', event => {
  const sweepId = event.target.value;
  loadRunIds(sweepId);
});

runsDropdown.addEventListener('change', event => {
  const sweepId = sweepsDropdown.value;
  const runId = event.target.value;
  loadData(sweepId, runId);
});

// Function to load observation images
async function loadObservationImages(sweepId, runId) {
  console.log('Loading observation images for sweep:', sweepId, 'run:', runId);
  try {
    // Load observation metadata
    const metadataResponse = await fetch(`html_data/${sweepId}/${runId}/observation_images_metadata.json`);
    if (!metadataResponse.ok) {
      console.log('No observation images metadata found');
      return;
    }
    
    const metadata = await metadataResponse.json();
    console.log('Loaded observation metadata:', metadata);
    appState.observationMetadata = metadata;
    
    // Load binary observation images
    const imagesResponse = await fetch(`html_data/${sweepId}/${runId}/observation_images.bin`);
    if (!imagesResponse.ok) {
      console.log('No observation images binary file found');
      return;
    }
    
    const imageBuffer = await imagesResponse.arrayBuffer();
    appState.observationImages = new Uint8Array(imageBuffer);
    console.log('Loaded observation images successfully');
  } catch (error) {
    console.error('Error loading observation images:', error);
  }
}

// Function to display observation image at a specific step
function displayObservationImage(stepIndex) {
  console.log('displayObservationImage called with stepIndex:', stepIndex);
  console.log('observationImages available:', !!appState.observationImages);
  console.log('observationMetadata available:', !!appState.observationMetadata);
  
  if (!appState.observationImages || !appState.observationMetadata || stepIndex >= appState.observationMetadata.length) {
    console.log('Cannot display observation image - missing data or invalid index');
    return;
  }
  
  const metadata = appState.observationMetadata[stepIndex];
  console.log('Image metadata:', metadata);
  const ctx = observationCanvas.getContext('2d');
  
  // Extract image data from binary buffer
  const startOffset = metadata.offset;
  const endOffset = startOffset + metadata.size;
  const imageData = appState.observationImages.slice(startOffset, endOffset);
  
  // Create ImageData object
  const width = metadata.width;
  const height = metadata.height;
  const channels = metadata.channels;
  
  // Fit image into container while maintaining aspect ratio
  const container = observationCanvas.parentElement;
  const containerWidth = container.clientWidth;
  const containerHeight = container.clientHeight;
  
  const aspectRatio = width / height;
  let scaledWidth, scaledHeight;
  
  if (containerWidth / containerHeight > aspectRatio) {
    // Container is wider than image aspect ratio
    scaledHeight = containerHeight;
    scaledWidth = Math.floor(containerHeight * aspectRatio);
  } else {
    // Container is taller than image aspect ratio
    scaledWidth = containerWidth;
    scaledHeight = Math.floor(containerWidth / aspectRatio);
  }
  
  // Calculate integer scale factor for pixel-perfect scaling
  const scaleX = Math.max(1, Math.floor(scaledWidth / width));
  const scaleY = Math.max(1, Math.floor(scaledHeight / height));
  const scale = Math.min(scaleX, scaleY);
  
  // Final canvas dimensions using integer scaling
  const finalWidth = width * scale;
  const finalHeight = height * scale;
  
  // Set canvas dimensions
  observationCanvas.width = finalWidth;
  observationCanvas.height = finalHeight;
  
  // Disable image smoothing for pixel-perfect scaling
  ctx.imageSmoothingEnabled = false;
  ctx.webkitImageSmoothingEnabled = false;
  ctx.mozImageSmoothingEnabled = false;
  ctx.msImageSmoothingEnabled = false;
  
  // Create scaled image data directly
  const scaledImageData = ctx.createImageData(finalWidth, finalHeight);
  
  // Handle different channel counts and scale pixels
  for (let y = 0; y < finalHeight; y++) {
    for (let x = 0; x < finalWidth; x++) {
      // Flip x coordinate for horizontal flip
      const flippedY = finalHeight - 1 - y;
      // Map scaled coordinates back to original image
      const origX = Math.floor(x / scale);
      const origY = Math.floor(flippedY / scale);
      const origIndex = origY * width + origX;
      const scaledIndex = y * finalWidth + x;
      let r, g, b;
      
      if (channels === 1) {
        // Grayscale (black and white)
        const grayValue = imageData[origIndex];
        r = g = b = grayValue;
      } else if (channels === 2) {
        // 2 channels - use first channel for RGB
        const value1 = imageData[origIndex * 2];
        r = g = b = value1;
      } else if (channels >= 3) {
        // RGB image (3 or more channels, use first 3)
        r = imageData[origIndex * channels];
        g = imageData[origIndex * channels + 1];
        b = imageData[origIndex * channels + 2];
      }
      
      // Set pixel in scaled image
      scaledImageData.data[scaledIndex * 4] = r;     // R
      scaledImageData.data[scaledIndex * 4 + 1] = g; // G
      scaledImageData.data[scaledIndex * 4 + 2] = b; // B
      scaledImageData.data[scaledIndex * 4 + 3] = 255; // A
    }
  }
  
  // Clear canvas and draw scaled image
  ctx.clearRect(0, 0, finalWidth, finalHeight);
  ctx.putImageData(scaledImageData, 0, 0);
}

function loadData(sweepId, runId) {
  // Store current sweep and run IDs in appState
  appState.currentSweepId = sweepId;
  appState.currentRunId = runId;
  
  // Reset bitmap layer cache when loading new data
  appState.currentBitmapLayer = null;
  appState.lastImagePath = null;
  
  // Pause the animation and reset the slider
  if (appState.isPlaying) {
    clearInterval(appState.playInterval);
    appState.isPlaying = false;
  }
  slider.value = 0;

  // Anchor point coordinates from parameters (longitude, latitude)
  const anchorPoint = [8.18362, 47.6014];

  fetch(`html_data/${sweepId}/${runId}/center_coordinates.json`)
    .then(response => {
      if (response.ok) return response.text();
      else throw new Error('Network response was not ok.');
    })
    .then(text => {
      const centerData = JSON.parse(text);
      const centerLongitude = centerData.center_lon;
      const centerLatitude = centerData.center_lat;

      // Load run configuration
      return fetch(`html_data/${sweepId}/${runId}/run_config.json`)
        .then(response => {
          if (response.ok) return response.json();
          else throw new Error('Network response was not ok.');
        })
        .then(config => {
          appState.runConfig = config;
          console.log('Loaded run config:', config);
          console.log('Radiation type:', config.radiation_type);
          console.log('All config keys:', Object.keys(config));
          console.log('Full config:', JSON.stringify(config, null, 2));
          
          // For 'jon' type, set map to hidden by default and hide the map button
          if (config.radiation_type === 'jon') {
            appState.showMap = false;
            const toggleMapButton = document.getElementById('toggleMapButton');
            if (toggleMapButton) {
              toggleMapButton.style.display = 'none';
            }
          }
          
          return { centerLongitude, centerLatitude };
        })
        .catch(error => {
          console.warn('Could not load run_config.json:', error);
          // Continue without config
          return { centerLongitude, centerLatitude };
        });
    })
    .then(({ centerLongitude, centerLatitude }) => {

      const initialViewState = {
        longitude: centerLongitude,
        latitude: centerLatitude,
        zoom: 11.5,
        pitch: 0,
        bearing: 0
      };
      // Mapbox access token
      const MAPBOX_TOKEN = window.MAPBOX_TOKEN;

      // Mapbox style id to human label mapping (for reference)
      const MAPBOX_STYLES = {
        'satellite-v9': 'Satellite',
        'streets-v12': 'Streets'
      };

      let currentMapStyle = 'satellite-v9';

      function getTextureUrl(style) {
        // Mapbox style id, e.g. 'satellite-v9', 'streets-v12'
        // Provide your own Mapbox access token via `window.MAPBOX_TOKEN`.
        return `https://api.mapbox.com/styles/v1/mapbox/${style}/tiles/256/{z}/{x}/{y}@2x?access_token=${MAPBOX_TOKEN}`;
      }
      const terrainLayer = new deck.TerrainLayer({
        id: 'terrain-layer',
        elevationData: 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png',
        texture: getTextureUrl(currentMapStyle),
        elevationDecoder: {
          rScaler: 256,
          gScaler: 1,
          bScaler: 1 / 256,
          offset: -32768
        },
        maxZoom: 15,
        operation: 'terrain+draw'
      });


      if (!appState.deckGL) {
        appState.deckGL = new deck.DeckGL({
          container,
          initialViewState,
          controller: {
            type: deck.MapController,
            maxPitch: 85
          },
          layers: [terrainLayer],
          onViewStateChange: ({viewState}) => {
            appState.currentViewState = viewState;
          }
        });
        // Initialize currentViewState with the initialViewState
        appState.currentViewState = initialViewState;
      } else {
        appState.deckGL.setProps({
          initialViewState,
          layers: [terrainLayer],
          onViewStateChange: ({viewState}) => {
            appState.currentViewState = viewState;
          }
        });
        // Update currentViewState with the new initialViewState
        appState.currentViewState = initialViewState;
      }

      // Load geo_json metadata first to understand the data structure
      fetch(`html_data/${sweepId}/${runId}/geo_json_metadata.json`)
        .then(response => {
          if (response.ok) return response.json();
          else throw new Error('Network response was not ok.');
        })
        .then(metadata => {
          console.log('Geo JSON metadata loaded:', metadata);
          
          // Initialize arrays based on metadata
          appState.paths = new Array(metadata.total_steps).fill(null);
          appState.totalMeasuredAreas = new Array(metadata.total_steps).fill(null);
          appState.colorsPerTimestep = new Array(metadata.total_steps).fill(null);
          appState.availableSteps = metadata.available_steps;
          
          slider.max = metadata.total_steps - 1;
          
          // Load metric_data.json
          fetch(`html_data/${sweepId}/${runId}/metric_data.json`)
                .then(response => {
                  if (response.ok) return response.text();
                  else throw new Error('Network response was not ok.');
                })
                .then(text => {
                  appState.metricData = JSON.parse(text);

                  // Try to load polygon data metadata (for z_rad_type == 'jon')
                  // Note: Polygon data is now loaded per-step, not all at once
                  
                  // First check if polygon data directory exists by trying to fetch a step
                  fetch(`html_data/${sweepId}/${runId}/polygon_data_chunks/step_0.json`)
                    .then(response => {
                      if (response.ok) {
                        // Polygon data exists, this is likely jon type
                        console.log('Polygon data detected - likely jon type radiation');
                        if (!appState.runConfig || appState.runConfig.radiation_type !== 'jon') {
                          // Set map visibility and hide button if not already set by config
                          appState.showMap = false;
                          const toggleMapButton = document.getElementById('toggleMapButton');
                          if (toggleMapButton) {
                            toggleMapButton.style.display = 'none';
                          }
                        }
                      }
                    })
                    .catch(error => {
                      console.log('No polygon data found, using point cloud visualization');
                    });
                  
                  appState.polygonData = new Array(metadata.total_steps).fill(null);
                  console.log('Polygon data will be loaded per-step for z_rad_type jon');

                  // Try to load PNG data (for z_rad_type == 'jon') - keeping for backward compatibility
                  // Note: PNG data is now loaded per-step, not all at once
                  appState.pngData = new Array(metadata.total_steps).fill(null);
                  console.log('PNG data will be loaded per-step for z_rad_type jon');

                  // Initialize point clouds array with empty data for each step
                  appState.pointClouds = new Array(metadata.total_steps).fill(null);
                  appState.pointCloudMetadata = null; // Store metadata for point cloud offsets
                  appState.pointCloudBuffer = null; // Store the full binary buffer

                  // Load point cloud metadata and binary data once
                  const metadataPath = `html_data/${sweepId}/${runId}/point_cloud_metadata.json`;
                  const binaryPath = `html_data/${sweepId}/${runId}/point_clouds.bin`;
                  
                  Promise.all([
                    fetch(metadataPath).then(response => {
                      if (response.ok) return response.json();
                      else throw new Error('Network response was not ok for metadata.');
                    }),
                    fetch(binaryPath).then(response => {
                      if (response.ok) return response.arrayBuffer();
                      else throw new Error('Network response was not ok for binary data.');
                    })
                  ])
                  .then(([metadata, arrayBuffer]) => {
                    appState.pointCloudMetadata = metadata;
                    appState.pointCloudBuffer = new Uint8Array(arrayBuffer);
                    console.log('Point cloud data loaded successfully');
                  })
                  .catch(error => {
                    console.warn('Point cloud data not available:', error);
                    appState.pointCloudMetadata = [];
                    appState.pointCloudBuffer = null;
                  });

                  // Helper function to render point cloud layer
                  function renderPointCloudLayer(index, baseLayers) {
                    let layers = [...baseLayers];
                    const pointCloudData = appState.pointClouds[index];
                    
                    if (pointCloudData && pointCloudData.length > 0) {
                      const pointCloudLayer = new deck.PointCloudLayer({
                        id: 'point-cloud-layer',
                        data: pointCloudData,
                        getPosition: d => d.position,
                        getColor: d => d.color,
                        pointSize: parseInt(bubbleSizeInput.value),
                        extensions: [
                          new deck._TerrainExtension({
                            terrainDrawMode: 'drape'
                          })
                        ]
                      });
                      layers.push(pointCloudLayer);
                    }
                    
                    appState.deckGL.setProps({
                      layers: layers
                    });
                    updateMetricInfo(index);
                  }
                  
                  // Helper function to update metric info
                  function updateMetricInfo(index) {
                    // Update metric info
                    metricContainer.innerHTML = '';
                    for (const [key, values] of Object.entries(appState.metricData)) {
                      const metricElement = document.createElement('div');
                      metricElement.className = 'metric-item';
                      metricElement.innerHTML = `<span>${key}:</span><span>${values[index]}</span>`;
                      metricContainer.appendChild(metricElement);
                    }
                  }

                  // Update layers function assigned to appState
                  appState.updateLayers = async function(index) {
                    // Load data for this step if not already loaded
                    await loadGeoJsonForStep(appState.currentSweepId, appState.currentRunId, index);
                    
                    // For 'jon' type, also load polygon data
                    const hasJonData = (appState.polygonData && appState.polygonData.length > 0) || 
                                       (appState.pngData && appState.pngData.length > 0);
                    const isJonType = (appState.runConfig && appState.runConfig.radiation_type === 'jon') || hasJonData;
                    
                    // If jon type is detected and map settings haven't been set yet, configure them
                    if (isJonType && appState.showMap !== false) {
                      appState.showMap = false;
                      const toggleMapButton = document.getElementById('toggleMapButton');
                      if (toggleMapButton) {
                        toggleMapButton.style.display = 'none';
                      }
                    }
                    
                    if (isJonType && !appState.polygonData[index]) {
                      await loadPolygonDataForStep(appState.currentSweepId, appState.currentRunId, index);
                    }
                    
                    // Path for the current index
                    const pathPoints = appState.paths[index];
                    // Process target_area depending on whether it is a multipolygon or not
                    let currentTotalMeasuredData = [];
                    const target = appState.totalMeasuredAreas[index];
                    if (target) {
                      let polygons = [];
                      // Check if target is actually a Polygon (an array of rings) or MultiPolygon (an array of polygons)
                      if (target.length > 0 && Array.isArray(target[0]) && target[0].length > 0 && Array.isArray(target[0][0])) {
                        if (typeof target[0][0][0] === 'number') {
                          // It is a Polygon: an array of rings, wrap it into an array
                          polygons = [target];
                        } else {
                          // It is a MultiPolygon: each element is an array of rings
                          polygons = target;
                        }
                      }
                      polygons.forEach(poly => {
                        currentTotalMeasuredData.push({ coordinates: poly });
                      });
                    }
                    // Create layers as before using the new currentTotalMeasuredData
                    const pathLayer = new deck.PathLayer({
                      id: 'path-layer',
                      data: [{ path: pathPoints }],
                      getPath: d => d.path,
                      getColor: hexToRgb(pathColorInput.value),
                      widthMinPixels: parseInt(pathSizeInput.value),
                      capRounded: false
                    });
                    const totalMeasuredAreaGeoJsonLayer = new deck.GeoJsonLayer({
                      id: 'geojson-layer',
                      data: {
                        type: 'FeatureCollection',
                        features: currentTotalMeasuredData.map(polygon => ({
                          type: 'Feature',
                          geometry: {
                            type: 'Polygon',
                            coordinates: polygon.coordinates
                          },
                          properties: {}
                        }))
                      },
                      getFillColor: [20, 225, 230, 100],
                      getLineColor: [255, 255, 255],
                      getLineWidth: 1,
                      stroked: false,
                      filled: true,
                      lineWidthMinPixels: 1,
                      // Only drape when map is visible
                      extensions: appState.showMap ? [
                        new deck._TerrainExtension({
                          terrainDrawMode: 'drape'
                        })
                      ] : []
                    });
                    
                    // Create base layers
                    let layers = [];
                    
                    // Add terrain layer only if showMap is true
                    if (appState.showMap) {
                      layers.push(terrainLayer);
                    }
                    
                    // Determine current image path
                    let currentImagePath = null;
                    let currentBounds = null;
                    
                    // Check if we have PNG data for this step (z_rad_type == 'jon')
                    if (appState.showBitmapLayer && appState.pngData && index < appState.pngData.length && appState.pngData[index]) {
                      const pngInfo = appState.pngData[index];
                      
                      let imagePath = pngInfo.path;
                      
                      currentImagePath = imagePath;
                      const originalBounds = pngInfo.bounds;
                      currentBounds = [
                        originalBounds[0], // right becomes left
                        originalBounds[3], // bottom stays the same
                        originalBounds[2], // left becomes right
                        originalBounds[1]  // top stays the same
                      ];
                      console.log(`Step ${index}: Found PNG data - ${currentImagePath}, showBitmapLayer: ${appState.showBitmapLayer}`);
                      
                      // Create bitmap layer (always recreate to ensure visibility changes work)
                      const bitmapLayer = new deck.BitmapLayer({
                        id: 'bitmap-layer',
                        image: currentImagePath,
                        bounds: currentBounds,
                        opacity: 0.8,
                        transparentColor: [255, 255, 255], // Make white pixels transparent
                        tintColor: [255, 255, 255], // Keep original colors
                        // Only drape when map is visible
                        extensions: appState.showMap ? [
                          new deck._TerrainExtension({
                            terrainDrawMode: 'drape'
                          })
                        ] : []
                      });
                      
                      layers.push(bitmapLayer);
                      console.log(`Step ${index}: Added bitmap layer to scene`);
                    } else if (appState.showBitmapLayer && appState.polygonData && index < appState.polygonData.length && appState.polygonData[index]) {
                      // Check if we have polygon data for this step (z_rad_type == 'jon' with new polygon format)
                      const polygonInfo = appState.polygonData[index];
                      
                      console.log(`Step ${index}: Found polygon data, showBitmapLayer: ${appState.showBitmapLayer}`);
                      
                      // Create GeoJSON layer for black polygons
                      const polygonLayer = new deck.GeoJsonLayer({
                        id: 'black-polygons-layer',
                        data: polygonInfo,
                        filled: true,
                        stroked: true,
                        getFillColor: [0, 0, 0, 180], // Black with some transparency
                        getLineColor: [0, 0, 0, 255], // Solid black border
                        getLineWidth: 2,
                        lineWidthMinPixels: 1,
                        // Only drape when map is visible
                        extensions: appState.showMap ? [
                          new deck._TerrainExtension({
                            terrainDrawMode: 'drape'
                          })
                        ] : []
                      });
                      
                      layers.push(polygonLayer);
                      console.log(`Step ${index}: Added polygon layer to scene`);
                    } else {
                      if (!appState.showBitmapLayer && ((appState.pngData && index < appState.pngData.length && appState.pngData[index]) || 
                                                       (appState.polygonData && index < appState.polygonData.length && appState.polygonData[index]))) {
                        console.log(`Step ${index}: Hidden state layer hidden by user (showBitmapLayer: ${appState.showBitmapLayer})`);
                      } else {
                        console.log(`Step ${index}: No hidden state data - index: ${index}, pngData length: ${appState.pngData ? appState.pngData.length : 'undefined'}, polygonData length: ${appState.polygonData ? appState.polygonData.length : 'undefined'}`);
                      }
                    }
                    
                    // Add other layers
                    layers.push(totalMeasuredAreaGeoJsonLayer, pathLayer);
                    
                    // Load point cloud data for current step if not already loaded
                    if (appState.showPointCloud) {
                      if (appState.pointClouds[index] === null) {
                        // Parse point cloud data from the consolidated binary file
                        if (appState.pointCloudBuffer && appState.pointCloudMetadata && 
                            index < appState.pointCloudMetadata.length) {
                          
                          const stepMeta = appState.pointCloudMetadata[index];
                          const numPoints = stepMeta.count;
                          
                          if (numPoints > 0) {
                            const startOffset = stepMeta.offset;
                            const endOffset = startOffset + (numPoints * 15); // 15 bytes per point
                            const stepData = appState.pointCloudBuffer.slice(startOffset, endOffset);
                            
                            // Parse binary data: 15 bytes per point (12 for xyz float32, 3 for rgb uint8)
                            const points = [];
                            
                            for (let i = 0; i < numPoints; i++) {
                              const offset = i * 15;
                              
                              // Extract position (3 * 4 bytes = 12 bytes for xyz as float32)
                              const posBytes = stepData.slice(offset, offset + 12);
                              const posFloats = new Float32Array(posBytes.buffer, posBytes.byteOffset, 3);
                              
                              // Extract color (3 bytes for rgb as uint8)
                              const r = stepData[offset + 12];
                              const g = stepData[offset + 13];
                              const b = stepData[offset + 14];
                              
                              points.push({
                                position: [posFloats[0], posFloats[1], posFloats[2]],
                                color: [r, g, b]
                              });
                            }
                            
                            appState.pointClouds[index] = points;
                          } else {
                            // No points for this step
                            appState.pointClouds[index] = [];
                          }
                        } else {
                          // No point cloud data available
                          appState.pointClouds[index] = [];
                        }
                      }
                      
                      // Render with point cloud
                      renderPointCloudLayer(index, layers);
                    } else {
                      // Point cloud disabled, just render without it
                      appState.deckGL.setProps({
                        layers: layers
                      });
                      updateMetricInfo(index);
                    }
                    
                    // Update observation image if visible
                    if (appState.showObservation) {
                      displayObservationImage(index);
                    }
                  };
                  
                  // Call updateLayers with initial index 0
                  appState.updateLayers(0);
                  
                  // Load observation images
                  loadObservationImages(sweepId, runId);
                })
                .catch(error => {
                  console.error('Error loading metric_data:', error);
                });
        })
        .catch(error => {
          console.error('Error loading geo_json_data:', error);
        });
    })
    .catch(error => {
      console.error('Error loading center coordinates:', error);
    });
}

function hexToRgb(hex) {
  const bigint = parseInt(hex.slice(1), 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = (bigint & 255);
  return [r, g, b];
}