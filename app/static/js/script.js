
let nodes = [];
let edges = [];
let nodeCounter = 0;  // Global variable to track the node count

let currentNodeId = 0;
let currentCheckpointId = null;  // To store the currently active checkpoint ID
let checkpointSelected = false;
let selectedNodes = [];  // Stack to store selected nodes
let contextNode = null; // Node related to the context menu

const container = document.getElementById('network-container');
let isDragging = false;
let dragNode = null;
let offsetX = 0;
let offsetY = 0;

function showPopup() {
    const popup = document.getElementById('popup');
    popup.style.display = 'block';
  }

// Function to hide the popup
function hidePopup() {
const popup = document.getElementById('popup');
popup.style.display = 'none';
}

  
document.addEventListener("DOMContentLoaded", function() {
const width = window.innerWidth;
const height = window.innerHeight;

if (width < 600 || height < 600) {
    showPopup();
} else {
    hidePopup();
}
});



window.addEventListener('resize', function() {
    const width = window.innerWidth;
    const height = window.innerHeight;

    if (width < 600 || height < 600) {
      showPopup();
    } else {
      hidePopup();
    }
    clearFCM();
    redrawFCM();  // Call the function to redraw the FCM whenever the window is resized
    resizeCanvas();
});

// Observe the container for any size changes
const resizeObserver = new ResizeObserver(entries => {
    for (let entry of entries) {
        // Trigger FCM redraw when the container is resized
        clearFCM();
        redrawFCM();  // Call the function to redraw the FCM whenever the window is resized
        resizeCanvas();
    }
});

// Start observing the container element
resizeObserver.observe(document.getElementById('network-container'));


function updateCheckpointState() {
    const currentCheckpoint = checkpoints.find(cp => cp.id === currentCheckpointId);
    
    if (currentCheckpoint) {
        // Update the nodes and edges of the current checkpoint
        currentCheckpoint.state.nodes = nodes.map(node => ({
            id: node.id,
            name: node.element.innerText,
            x: node.element.style.left.replace('px', '') / container.offsetWidth,
            y: node.element.style.top.replace('px', '') / container.offsetHeight
        }));
        
        currentCheckpoint.state.edges = edges.map(edge => ({
            from: edge.from.id,
            to: edge.to.id,
            value: edge.value,
            color: edge.color
        }));

        console.log(`Checkpoint ${currentCheckpointId} updated successfully.`);
    } else {
        console.error(`Checkpoint ${currentCheckpointId} not found.`);
    }
}



// Add event listener for keydown to handle shortcuts
document.addEventListener('keydown', function (event) {
    // Make sure we detect only certain combinations
    if (event.ctrlKey && !event.altKey && !event.shiftKey) {
        switch (event.key.toLowerCase()) {
            case 's':  // Ctrl + S for "Save Project"
                event.preventDefault();  // Prevent browser's default Ctrl + S save behavior
                triggerButton('save-btn');
                break;
            case 'l':  // Ctrl + L for "Load Project"
                event.preventDefault();
                triggerButton('load-btn');
                break;
            case 'q':  // Ctrl + X for "Add Node"
                event.preventDefault();
                triggerButton('add-node-btn');
                break;
            case 'e':  // Ctrl + E for "Add Edge"
                event.preventDefault();
                triggerButton('add-edge-btn');
                break;
        }
    }
});



const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');
container.appendChild(canvas);

canvas.width = container.offsetWidth;
canvas.height = container.offsetHeight;

async function saveProject() {
    if (!sessionPassword) {
        alert('Please set a session password to encrypt the project.');
        return;
    }

    // Store nodes and edges in fcmState
    fcmState.nodes = nodes.map(node => ({
        id: node.id,
        name: node.element.innerText,
        x: node.element.style.left.replace('px', '') / container.offsetWidth,  // Store relative positions
        y: node.element.style.top.replace('px', '') / container.offsetHeight
    }));

    fcmState.edges = edges.map(edge => ({
        from: edge.from.id,
        to: edge.to.id,
        value: edge.value,
        color: edge.color
    }));

    // Include checkpoints in fcmState before saving
    fcmState.checkpoints = checkpoints.map(checkpoint => ({
        id: checkpoint.id,
        name: checkpoint.name,
        timestamp: checkpoint.timestamp,
        state: checkpoint.state
    }));

    // Convert project state to JSON
    const projectData = JSON.stringify(fcmState);

    try {
        // Encrypt the project data with the session password
        const encryptedBlob = await encryptData(sessionPassword, projectData);

        // Create a download link for the encrypted project file
        const link = document.createElement('a');
        link.href = URL.createObjectURL(encryptedBlob);
        link.download = 'fcm_project.fcmvss';  // Set file name for the project
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        console.log('Project saved and encrypted successfully.');
    } catch (error) {
        console.error('Error during encryption:', error);
        alert('Failed to save the project.');
    }
}

async function loadProject(files) {
    if (files.length === 0) {
        alert('No file selected.');
        return;
    }

    const file = files[0];
    const reader = new FileReader();

    reader.onload = async function (event) {
        try {
            const encryptedBlob = new Blob([event.target.result]);

            // Step 1: Try decryption with session password
            if (sessionPassword) {
                try {
                    const decryptedData = await decryptData(sessionPassword, encryptedBlob);
                    loadProjectData(JSON.parse(decryptedData));  // Proceed if decryption with session password succeeds
                    return;  // Exit if session password works
                } catch (error) {
                    console.warn('Session password decryption failed.Kindly save the current work and reload the app to retain importing/exporting functionalities, as they are intentionally temporarily comprised for safety reasons.');
                     // Fail-safe: Reload window on failure
                }
            }

            // Step 2: If session password fails, show the password popup to ask the user for password
            showPasswordPopup(async function (manualPassword) {
                if (!manualPassword || manualPassword.trim() === '') {
                    alert('You must enter a valid password.');
                    return;
                }

                try {
                    const decryptedData = await decryptData(manualPassword, encryptedBlob);
                    loadProjectData(JSON.parse(decryptedData));  // Proceed if decryption with manual password succeeds
                } catch (error) {
                    console.error('Manual password decryption failed.');
                    alert('Failed to load the project. Please ensure the password is correct and the file is valid. Kindly save the current work (if performed) and reload the app to retain importing/exporting functionalities, as they are intentionally temporarily comprised for safety reasons.');
                     // Fail-safe: Reload window on failure
                }
            });

        } catch (error) {
            console.error('Error during project load:', error);
            alert('Failed to load the project. Kindly save the current work (if performed) and reload the app to retain importing/exporting functionalities, as they are intentionally temporarily comprised for safety reasons.');
        }
    };

    reader.readAsArrayBuffer(file);
}

// Helper function to load project data
function loadProjectData(loadedState) {
    if (!loadedState.nodes || !loadedState.edges) {
        alert('Invalid project file. No nodes or edges found.');
        return;
    }

    clearFCM();  // Clear existing nodes and edges
    fcmState.nodes = loadedState.nodes;
    fcmState.edges = loadedState.edges;

    redrawFCM();  // Redraw nodes and edges
    if (loadedState.checkpoints) {
        checkpoints = loadedState.checkpoints;
        const tray = document.getElementById('checkpoint-tray');
        tray.innerHTML = '';  // Clear current checkpoint tray

        checkpoints.forEach(checkpoint => displayCheckpointButton(checkpoint));
        checkpointCounter = Math.max(...checkpoints.map(cp => cp.id)) + 1;
    } else {
        checkpointCounter = 1;  // Reset if no checkpoints exist
    }

    // **Update the tables and chart after loading the project**
    updateNodeStats();
    console.log('Project loaded successfully.');
}


// Function to resize the canvas based on the container's size
function resizeCanvas() {
canvas.width = container.offsetWidth;
canvas.height = container.offsetHeight;
updateEdges();  // Redraw edges when the canvas size changes
}
function drawArrowHead(ctx, x, y, angle, length, color) {
const arrowAngle = Math.PI / 6; // 30-degree angle for the arrowhead

ctx.beginPath();
ctx.moveTo(x, y);
ctx.lineTo(
    x - length * Math.cos(angle - arrowAngle),
    y - length * Math.sin(angle - arrowAngle)
);
ctx.moveTo(x, y);
ctx.lineTo(
    x - length * Math.cos(angle + arrowAngle),
    y - length * Math.sin(angle + arrowAngle)
);
ctx.strokeStyle = color;
ctx.stroke();
}


// Function to draw continuous arrows along the edge
// Function to draw continuous arrows along the edge
function drawContinuousArrows(ctx, fromX, fromY, toX, toY, value, color) {
const arrowSpacing = 30; // Distance between each arrow along the line
const arrowLength = 10;  // Length of each arrow

const dx = toX - fromX;
const dy = toY - fromY;
const distance = Math.sqrt(dx * dx + dy * dy);
const steps = Math.floor(distance / arrowSpacing); // Number of arrows to draw along the line
const angle = Math.atan2(dy, dx); // Angle of the line

// Draw the main line
ctx.beginPath();
ctx.strokeStyle = color;
ctx.moveTo(fromX, fromY);
ctx.lineTo(toX, toY);
ctx.stroke();

// Draw repeated arrows along the line
for (let i = 0; i < steps; i++) {
    const arrowX = fromX + (i * arrowSpacing) * Math.cos(angle);
    const arrowY = fromY + (i * arrowSpacing) * Math.sin(angle);
    drawArrowHead(ctx, arrowX, arrowY, angle, arrowLength, color);
}

// Draw the edge value at the midpoint of the line
const midX = (fromX + toX) / 2;
const midY = (fromY + toY) / 2;
ctx.font = "14px Arial";
ctx.fillStyle = "black";
ctx.fillText(value, midX, midY - 10);
}



// Ensure the canvas is resized correctly when loading
window.onload = function () {
resizeCanvas();
};
const increaseBtn = document.getElementById('increase-btn');
const decreaseBtn = document.getElementById('decrease-btn');

// Function to increase height by 10%
increaseBtn.addEventListener('click', () => {
    // Get the current height in pixels and add 10%
    if(!checkpointSelected){
        alert('Consider saving and then selecting your FCM as a checkpoint before altering dimensions of playground. Otherwise edges would not be registered')
        return;
    }
    const currentHeight = container.offsetHeight;
    const newHeight = currentHeight * 1.1; // Increase by 10%
    container.style.height = `${newHeight}px`;

    // Adjust node positions relative to the new height
    nodes.forEach(node => {
        const nodeState = fcmState.nodes.find(n => n.id === node.id);
        if (nodeState) {
            node.element.style.left = `${nodeState.x * container.offsetWidth}px`;
            node.element.style.top = `${nodeState.y * container.offsetHeight}px`;
        }
    });
    redrawEdges();
});

// Function to reduce height by 10%
decreaseBtn.addEventListener('click', () => {
    if(!checkpointSelected){
        alert('Consider saving and then selecting your FCM as a checkpoint before altering dimensions of playground. Otherwise edges would not be registered')
        return;
    }
    // Get the current height in pixels and subtract 10%
    const currentHeight = container.offsetHeight;
    const newHeight = currentHeight * 0.9; // Decrease by 10%
    container.style.height = `${newHeight}px`;

    // Adjust node positions relative to the new height
    nodes.forEach(node => {
        const nodeState = fcmState.nodes.find(n => n.id === node.id);
        if (nodeState) {
            node.element.style.left = `${nodeState.x * container.offsetWidth}px`;
            node.element.style.top = `${nodeState.y * container.offsetHeight}px`;
        }
    });
    redrawEdges();
});

// Function to resize the canvas based on the container's size
function resizeCanvas() {
    canvas.width = container.offsetWidth;
    canvas.height = container.offsetHeight;
    updateEdges();  // Ensure edges are updated whenever the canvas is resized
}

// Function to handle node click selection
container.addEventListener('click', (e) => {
    const clickedNode = e.target.closest('.node');  // Get the node, even if the text is clicked

    if (clickedNode) {
        const nodeId = parseInt(clickedNode.getAttribute('data-id'));
        const existingNodeIndex = selectedNodes.findIndex(n => n.id === nodeId);

        // If the node is already selected, deselect it
        if (existingNodeIndex !== -1) {
            clickedNode.style.backgroundColor = '#5482be';  // Default color for deselected nodes
            selectedNodes.splice(existingNodeIndex, 1);  // Remove node from selectedNodes
        } 
        // If fewer than 2 nodes are selected, select this node
        else if (selectedNodes.length < 2) {
            if (selectedNodes.length === 0) {
                clickedNode.style.backgroundColor = 'blue';  // Color the first node blue
            } else {
                clickedNode.style.backgroundColor = 'purple';  // Color the second node purple
            }
            selectedNodes.push({ id: nodeId, element: clickedNode });  // Add node to the selectedNodes array
        } 
        // If both nodes are already selected, reset the selection and start again
        else if (selectedNodes.length === 2) {
            // Deselect both currently selected nodes
            selectedNodes[0].element.style.backgroundColor = '#5482be';
            selectedNodes[1].element.style.backgroundColor = '#5482be';

            // Clear the selection and select the new node
            selectedNodes = [];
            clickedNode.style.backgroundColor = 'blue';
            selectedNodes.push({ id: nodeId, element: clickedNode });
        }
    }
});

// Function to calculate the intersection point between the line and the rectangular node's edge
function getRectIntersection(fromX, fromY, toX, toY, halfWidth, halfHeight) {
    const dx = toX - fromX;
    const dy = toY - fromY;
    let endX = toX, endY = toY;

    // Slopes and ratios to determine which edge of the rectangle the line intersects
    if (Math.abs(dx) > Math.abs(dy)) {
        // Hits left or right side
        const slopeY = (dy / dx) * halfWidth;
        if (dx > 0) {
            // Hit right side
            endX = toX - halfWidth;
            endY = toY - slopeY;
        } else {
            // Hit left side
            endX = toX + halfWidth;
            endY = toY + slopeY;
        }
    } else {
        // Hits top or bottom
        const slopeX = (dx / dy) * halfHeight;
        if (dy > 0) {
            // Hit bottom side
            endX = toX - slopeX;
            endY = toY - halfHeight;
        } else {
            // Hit top side
            endX = toX + slopeX;
            endY = toY + halfHeight;
        }
    }
    return { endX, endY };
}

// Function to draw a directed edge (arrow) between two points and display the edge value
function drawArrow(ctx, fromX, fromY, toX, toY, value, color) {
    const headLength = 10; // Length of the arrowhead
    const angle = Math.atan2(toY - fromY, toX - fromX);

    // Draw line from the starting node to the adjusted point (before the receiving node's edge)
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.moveTo(fromX, fromY);
    ctx.lineTo(toX, toY);
    ctx.stroke();

    // Draw the arrowhead
    ctx.beginPath();
    ctx.moveTo(toX, toY);
    ctx.lineTo(toX - headLength * Math.cos(angle - Math.PI / 6), toY - headLength * Math.sin(angle - Math.PI / 6));
    ctx.moveTo(toX, toY);
    ctx.lineTo(toX - headLength * Math.cos(angle + Math.PI / 6), toY - headLength * Math.sin(angle + Math.PI / 6));
    ctx.stroke();

    // Draw the edge value at the midpoint of the line
    const midX = (fromX + toX) / 2;
    const midY = (fromY + toY) / 2;
    ctx.font = "14px Arial";
    ctx.fillStyle = "black";
    ctx.fillText(value, midX, midY - 10);
}

// Function to calculate distance from a point to a line
function pointToLineDistance(x, y, x1, y1, x2, y2) {
    const A = x - x1;
    const B = y - y1;
    const C = x2 - x1;
    const D = y2 - y1;

    const dot = A * C + B * D;
    const len_sq = C * C + D * D;
    const param = len_sq !== 0 ? dot / len_sq : -1;

    let xx, yy;

    if (param < 0) {
        xx = x1;
        yy = y1;
    } else if (param > 1) {
        xx = x2;
        yy = y2;
    } else {
        xx = x1 + param * C;
        yy = y1 + param * D;
    }

    const dx = x - xx;
    const dy = y - yy;
    return Math.sqrt(dx * dx + dy * dy);
}

window.addEventListener('resize', () => {
    canvas.width = container.offsetWidth;
    canvas.height = container.offsetHeight;
    updateEdges();
});
// Function to remove a node and its associated edges

// Helper function to clean the node name by removing any trailing X or R
function cleanNodeName(name) {
    return name.trim().replace(/\s*X\s*$/, '').replace(/\s*R\s*$/, '').replace(/\s*X\s*$/, '').replace(/\s*R\s*$/, '');  // Remove X and R at the end of the name
}

// Example function to show the Kosko results table when content is added
function showKoskoResultsTable() {
    const koskoTable = document.getElementById('kosko-results-table');
    if (koskoTable) {
        koskoTable.style.display = 'table'; // Make the table visible
    }
}

// Example function to show the fixation results table when content is added
function showFixationResultsTable() {
    const fixationTable = document.getElementById('fixation-results-table');
    if (fixationTable) {
        fixationTable.style.display = 'table'; // Make the table visible
    }
}

// Example function to show the Kosko line chart when content is added
function showKoskoLineChart() {
    const koskoChart = document.getElementById('kosko-line-chart');
    if (koskoChart) {
        koskoChart.style.display = 'block'; // Make the chart visible
    }
}

// Example function to show the difference bar chart when content is added
function showDifferenceBarChart() {
    const diffChart = document.getElementById('difference-bar-chart');
    if (diffChart) {
        diffChart.style.display = 'block'; // Make the chart visible
    }
}

// When you populate the tables and charts with data, call these functions.


function removeNode() {
    if (contextNode) {
        const nodeId = parseInt(contextNode.getAttribute('data-id'));
        let nodeName = cleanNodeName(contextNode.innerText);  // Clean the node name here

        // Remove node element from the DOM
        container.removeChild(contextNode);

        // Remove the node from the nodes array
        nodes = nodes.filter(node => node.id !== nodeId);

        // Find all edges associated with the node
        const removedEdges = edges.filter(edge => edge.from.id === nodeId || edge.to.id === nodeId);
        
        // Remove all edges associated with the node
        edges = edges.filter(edge => edge.from.id !== nodeId && edge.to.id !== nodeId);

        // Update the edges on the canvas
        updateEdges();

        // Log the node removal in the timeline
        logChange(`Deleted node: ${cleanNodeName(nodeName)}`);  // Use the cleaned node name

        // Log the associated edges removal in the timeline
        removedEdges.forEach(edge => {
            const fromNodeName = cleanNodeName(edge.from.element.innerText);  // Ensure "XR" is removed
            const toNodeName = cleanNodeName(edge.to.element.innerText);  // Ensure "XR" is removed
            logChange(`Removed edge between ${fromNodeName} and ${toNodeName}`);
        });
        updateNodeStats();
    }
}




function showModal(promptText, callback) {
const modal = document.getElementById('input-modal');
const label = document.getElementById('modal-label');
const input = document.getElementById('modal-input');

label.innerText = promptText;
input.value = ''; // Clear previous input
modal.style.display = 'flex'; // Show modal

// Handle the OK button
document.getElementById('modal-ok-btn').onclick = () => {
    const value = input.value;
    modal.style.display = 'none'; // Hide modal
    callback(value); // Return the value to the callback function
};

// Handle the Cancel button
document.getElementById('modal-cancel-btn').onclick = () => {
    modal.style.display = 'none'; // Hide modal
};
}
document.getElementById('modal-cancel-btn').onclick = () => {
document.getElementById('input-modal').style.display = 'none';
};

// FCM state storage
let fcmState = {
nodes: [],
edges: []
};

let checkpoints = [];
let checkpointCounter = 1;


let sessionPassword = null;  // To store the session password temporarily

function setSessionPassword() {
    const password = prompt("Please enter your session password:");
    
    if (!password || password.trim() === '') {
        alert('You must enter a valid password to proceed.');
        return;
    }

    sessionPassword = password;  // Store the password in the sessionPassword variable
    alert('Session password set successfully!');
}


function addEdge() {
    if (selectedNodes.length !== 2) {
        alert('Please select exactly two nodes to connect.');
        return;
    }

    const selectedNode1 = selectedNodes[0];
    const selectedNode2 = selectedNodes[1];

    const edgeValue = prompt("Please enter a value for this edge (non-zero):", "1");
    if (edgeValue === null || parseFloat(edgeValue) === 0) {
        alert("Edge value cannot be zero.");
        return;
    }
    else if (edgeValue > 1 || edgeValue < -1) {
        alert("Set Edge value only within [-1,1] inclusively");
        return;
    }

    const edgeColor = parseFloat(edgeValue) < 0 ? 'red' : 'green';

    edges.push({
        from: selectedNode1,
        to: selectedNode2,
        value: edgeValue,
        color: edgeColor
    });

    updateEdges();  

    // Use cleanNodeName to remove X and R from node names before logging
    logChange(`Added edge between ${cleanNodeName(selectedNode1.element.innerText)} and ${cleanNodeName(selectedNode2.element.innerText)}`);

    selectedNode1.element.style.backgroundColor = '#5482be';
    selectedNode2.element.style.backgroundColor = '#5482be';
    selectedNodes = [];
    updateNodeStats();
}

function saveCheckpoint(checkpointName = null) {
    const clonedNodes = nodes.map(node => ({
        id: node.id,
        name: node.element.innerText,
        x: node.element.style.left.replace('px', '') / container.offsetWidth,  // Relative positions
        y: node.element.style.top.replace('px', '') / container.offsetHeight  // Relative positions
    }));

    const clonedEdges = edges.map(edge => ({
        from: edge.from.id,
        to: edge.to.id,
        value: edge.value,
        color: edge.color
    }));

    const checkpoint = {
        id: checkpointCounter,  // Increment the checkpoint counter
        state: {
            nodes: clonedNodes,
            edges: clonedEdges
        },
        name: checkpointName || `CKP ${checkpointCounter}`,  // Default or custom name
        timestamp: new Date().toISOString()  // Timestamp for reference
    };
    checkpointCounter++;
    // Add the checkpoint to the checkpoints array
    checkpoints.push(checkpoint);

    // Display the checkpoint button in the tray
    displayCheckpointButton(checkpoint);
    logChange(`Saved Checkpoint ${checkpoint.name} successfully`);
    console.log(`${checkpoint.name} saved successfully.`);
}


function displayCheckpointButton(checkpoint) {
    const tray = document.getElementById('checkpoint-tray');
    const buttonWrapper = document.createElement('div');
    
    // Wrapper styling to contain both checkpoint button and "X" button
    buttonWrapper.style.position = 'relative';
    buttonWrapper.style.display = 'inline-block';
    buttonWrapper.style.marginBottom = '10px'; // Add spacing between buttons

    // Create the main checkpoint button
    const button = document.createElement('button');
    button.textContent = `${checkpoint.name} - ${new Date(checkpoint.timestamp).toLocaleString()}`;
    button.setAttribute('data-id', checkpoint.id);

    // Restore the checkpoint when clicked
    button.onclick = function () {
        restoreCheckpoint(checkpoint.id);
    };

    // Create the small "X" button for removing the checkpoint
    const removeBtn = document.createElement('button');
    removeBtn.innerHTML = '&times;'; // HTML code for '×' symbol
    removeBtn.style.position = 'absolute';
    removeBtn.style.backgroundColor = '#ff5252';
    removeBtn.style.color = 'white';
    removeBtn.style.border = 'none';
    removeBtn.style.padding = '5px';
    removeBtn.style.fontSize = '12px';
    removeBtn.style.width = '25px';
    removeBtn.style.height = '25px';
    removeBtn.style.borderRadius = '50%';
    removeBtn.style.cursor = 'pointer';
    removeBtn.style.boxShadow = '0 2px 5px rgba(0, 0, 0, 0.2)';
    removeBtn.style.transition = 'background-color 0.3s ease, transform 0.3s ease';

    // Hover effect for the remove button
    removeBtn.onmouseover = function () {
        removeBtn.style.backgroundColor = '#e53935';
        removeBtn.style.transform = 'scale(1.1)';
    };

    removeBtn.onmouseout = function () {
        removeBtn.style.backgroundColor = '#ff5252';
        removeBtn.style.transform = 'scale(1)';
    };

    // Attach click event to remove the checkpoint
    removeBtn.onclick = function (event) {
        event.stopPropagation(); // Prevent the main button's click event
        removeCheckpoint(checkpoint.id, buttonWrapper);
    };

    // Append the button and the remove button to the wrapper
    buttonWrapper.appendChild(button);
    buttonWrapper.appendChild(removeBtn);

    // Append the wrapper to the tray
    tray.appendChild(buttonWrapper);
    updateNodeStats()
}


function removeCheckpoint(checkpointId, buttonElement) {
    // Find the index of the checkpoint
    const checkpointIndex = checkpoints.findIndex(cp => cp.id === checkpointId);
    
    if (checkpointIndex !== -1) {
        // Remove the checkpoint from the array
        checkpoints.splice(checkpointIndex, 1);
        console.log(`Checkpoint ${checkpointId} deleted successfully.`);
        logChange(`Deleted Checkpoint ${checkpointId} successfully`);
        checkpointCounter--;
        // Remove the button from the UI
        buttonElement.remove();
    } else {
        console.error(`Checkpoint with id ${checkpointId} not found.`);
    }
}

document.getElementById('photo').addEventListener('click', function() {
    const networkContainer = document.getElementById('network-container');
  
    // Use html2canvas to capture the container as an image
    html2canvas(networkContainer).then(function(canvas) {
      const context = canvas.getContext('2d');
  
      // Get the current date and time
      const currentDate = new Date();
      const dateTimeString = `Captured by FCM-VSS (${currentDate.toLocaleString()})`;
  
      // Set font style for the text
      context.font = '20px Arial';
      context.fillStyle = 'black'; // Text color
  
      // Add some padding to the bottom for the text
      const textHeight = 30; // Height to reserve for the text
      const canvasWithText = document.createElement('canvas');
      canvasWithText.width = canvas.width;
      canvasWithText.height = canvas.height + textHeight;
  
      // Draw the original canvas content on the new canvas
      const newContext = canvasWithText.getContext('2d');
      newContext.drawImage(canvas, 0, 0);
  
      // Add the text at the bottom
      newContext.fillText(dateTimeString, 10, canvas.height + 20);
  
      // Convert the new canvas with text to a data URL (base64 image)
      const imgData = canvasWithText.toDataURL('image/png');
  
      // Create a temporary link element to download the image
      const link = document.createElement('a');
      link.href = imgData;
      link.download = 'network_image.png'; // Filename for the downloaded image
  
      // Trigger the download by programmatically clicking the link
      link.click();
    }).catch(function(error) {
      console.error('Error capturing the image:', error);
    });
  });
  

  function displayCheckpointName(name) {
    const checkpointDisplay = document.getElementById('checkpoint-name-display');
    if (checkpointDisplay) {
        checkpointDisplay.innerText = name;  // Display the correct checkpoint name
        checkpointDisplay.style.color = '#5482be';  // Change the color to blue when the name is injected
    }
}


function redrawEdges() {
    fcmState.edges.forEach(edgeState => {
        const fromNode = nodes.find(n => n.id == edgeState.from);
        const toNode = nodes.find(n => n.id == edgeState.to);

        if (fromNode && toNode) {
            edges.push({
                from: fromNode,
                to: toNode,
                value: edgeState.value,
                color: edgeState.color
            });
        }
    });
    // Redraw edges on the canvas
    updateEdges();
    updateNodeStats()
    
}


let selectedCheckpointId = null;  // Store the currently selected checkpoint ID
function restoreCheckpoint(id) {
    const checkpoint = checkpoints.find(cp => cp.id === id);
    if (checkpoint) {
        // Clear the current FCM state and visual elements
        clearFCM();

        // Clone the saved state from the checkpoint
        const restoredNodes = checkpoint.state.nodes.map(node => ({ ...node }));
        const restoredEdges = checkpoint.state.edges.map(edge => ({ ...edge }));

        // Update the current FCM state with the restored state
        fcmState.nodes = restoredNodes;
        fcmState.edges = restoredEdges;

        // First, redraw nodes from the restored state
        redrawNodes();

        // After nodes are rendered, redraw edges based on node positions
        setTimeout(() => {
            redrawEdges();
        }, 100);  // Add a small delay to ensure nodes are rendered before edges

        console.log(`Checkpoint ${id} restored successfully.`);

        // Highlight the selected checkpoint button
        highlightCheckpointButton(id);

        // Display the checkpoint name or timestamp in the UI
        displayCheckpointName(checkpoint.name);
        checkpointSelected = true;
        // **Update the tables and the chart after restoring the checkpoint**
        updateNodeStats();  // Ensure node stats are updated
          // Ensure the Kosko chart is updated
    } else {
        console.error(`Checkpoint with id ${id} not found.`);
    }
}

function redrawNodes() {
    fcmState.nodes.forEach(nodeState => {
        const node = document.createElement('div');
        node.classList.add('node');
        node.innerText = nodeState.name;

        // Restore node's position based on its saved relative coordinates
        node.style.left = `${nodeState.x * container.offsetWidth}px`;
        node.style.top = `${nodeState.y * container.offsetHeight}px`;
        node.setAttribute('data-id', nodeState.id);

        // Create the "X" (Remove) button
        const removeBtn = document.createElement('button');
        removeBtn.classList.add('node-remove-btn');
        removeBtn.innerText = 'X';
        removeBtn.style.position = 'absolute';
        removeBtn.style.right = '0px';  // Align on the right of the node
        removeBtn.style.top = '0px';    // Align at the top of the node

        // Create the "R" (Rename) button
        const renameBtn = document.createElement('button');
        renameBtn.classList.add('node-rename-btn');
        renameBtn.innerText = 'R';
        renameBtn.style.position = 'absolute';
        renameBtn.style.left = '0px';   // Align on the left of the node
        renameBtn.style.top = '0px';    // Align at the top of the node

        // Append buttons to the node
        node.appendChild(removeBtn);
        node.appendChild(renameBtn);

        // Add the node to the container
        container.appendChild(node);
        nodes.push({ id: nodeState.id, element: node });

        // Attach event listeners to make the node interactive again
        attachNodeEventListeners(node);
    });
}



function clearFCM() {
// Remove all nodes from the DOM
nodes.forEach(node => container.removeChild(node.element));
nodes = [];  // Clear the nodes array

// Clear the edges array and the canvas
edges = [];
ctx.clearRect(0, 0, canvas.width, canvas.height);
}


// Initial checkpoint setup on page load
window.onload = function () {
saveCheckpoint();
};


function redrawFCM() {
// Restore nodes from the fcmState
fcmState.nodes.forEach(nodeState => {
    const node = document.createElement('div');
    node.classList.add('node');
    node.innerText = nodeState.name;  // Set the node's name

    // Restore node's position based on its saved relative coordinates
    node.style.left = `${nodeState.x * container.offsetWidth}px`;
    node.style.top = `${nodeState.y * container.offsetHeight}px`;
    node.setAttribute('data-id', nodeState.id);

    container.appendChild(node);
    nodes.push({ id: nodeState.id, element: node });

    // Attach event listeners to make the node interactive again
    attachNodeEventListeners(node);

});

// Restore edges from the fcmState
fcmState.edges.forEach(edgeState => {
    const fromNode = nodes.find(n => n.id == edgeState.from);
    const toNode = nodes.find(n => n.id == edgeState.to);

    if (fromNode && toNode) {
        edges.push({
            from: fromNode,
            to: toNode,
            value: edgeState.value,
            color: edgeState.color
        });
    }
});
// Redraw edges on the canvas
updateEdges();
}

// Function to update edges
function updateEdges() {
    // Clear the canvas to redraw the edges
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Remove any existing "X" buttons before redrawing
    const existingEdgeButtons = document.querySelectorAll('.edge-remove-btn');
    existingEdgeButtons.forEach(btn => btn.remove());

    edges.forEach((edge, index) => {
        const fromNode = edge.from.element;
        const toNode = edge.to.element;

        const fromX = fromNode.offsetLeft + fromNode.offsetWidth / 2;
        const fromY = fromNode.offsetTop + fromNode.offsetHeight / 2;
        const toX = toNode.offsetLeft + toNode.offsetWidth / 2;
        const toY = toNode.offsetTop + toNode.offsetHeight / 2;

        // Draw the edge with continuous arrows
        drawContinuousArrows(ctx, fromX, fromY, toX, toY, edge.value, edge.color);

        // Calculate the midpoint of the edge
        const midX = (fromX + toX) / 2;
        const midY = (fromY + toY) / 2;

        // Create a small "X" button at the midpoint
        const removeBtn = document.createElement('button');
        removeBtn.classList.add('edge-remove-btn');
        removeBtn.innerText = 'X';
        removeBtn.style.left = `${midX - 10}px`;  // Position the button at the midpoint
        removeBtn.style.top = `${midY - 10}px`;
        removeBtn.style.position = 'absolute';
        removeBtn.style.width = '20px';  // Small button size
        removeBtn.style.height = '20px';
        removeBtn.style.fontSize = '12px';
        removeBtn.style.cursor = 'pointer';
        removeBtn.style.backgroundColor = 'red';  // Red "X" button for visibility
        removeBtn.style.borderRadius = '50%';  // Make it circular
        removeBtn.style.border = 'none';
        removeBtn.style.color = 'white';

        // Add click event to remove the edge
        removeBtn.addEventListener('click', () => {
            edges.splice(index, 1);  // Remove the clicked edge from the edges array
            updateEdges();  // Redraw the edges after removal
            updateNodeStats();
        });

        // Append the "X" button to the container
        container.appendChild(removeBtn);
    });
}


// Function to attach event listeners for renaming, removing, and dragging nodes
function attachNodeEventListeners(node) {
    const nodeId = parseInt(node.getAttribute('data-id'));

    // Create the "X" (Remove) button
    const removeBtn = document.createElement('button');
    removeBtn.classList.add('node-remove-btn');
    removeBtn.innerText = 'X';
    removeBtn.style.position = 'absolute';
    removeBtn.style.right = '0px';  // Align on the right of the node
    removeBtn.style.top = '0px';    // Align at the top of the node

    // Create the "R" (Rename) button
    const renameBtn = document.createElement('button');
    renameBtn.classList.add('node-rename-btn');
    renameBtn.innerText = 'R';
    renameBtn.style.position = 'absolute';
    renameBtn.style.left = '0px';   // Align on the left of the node
    renameBtn.style.top = '0px';    // Align at the top of the node

    // Append buttons to the node
    node.appendChild(removeBtn);
    node.appendChild(renameBtn);

    // Event listener to remove node
    removeBtn.addEventListener('click', () => {
        container.removeChild(node);  // Remove the node element from the DOM
        nodes = nodes.filter(n => n.id !== nodeId);  // Remove from nodes array
        edges = edges.filter(edge => edge.from.id !== nodeId && edge.to.id !== nodeId);  // Remove edges associated with the node
        updateEdges();  // Redraw edges without the removed node
        logChange(`Deleted node: ${cleanNodeName(node.innerText.slice(0, -1))}`);
        updateNodeStats();
        
    });

    // Event listener to rename node
// Event listener to rename node
renameBtn.addEventListener('click', () => {
    showModal("Enter the new name for this node:", (newName) => {
        if (newName !== null && newName.trim() !== "") {
            const oldName = cleanNodeName(node.innerText);  // Clean the old name

            // Clear existing text and buttons
            while (node.firstChild) {
                node.removeChild(node.firstChild); // Clear existing content
            }

            const nodeText = document.createElement('span');
            nodeText.innerText = cleanNodeName(newName);  // Clean the new name
            node.appendChild(nodeText);
            node.appendChild(renameBtn);
            node.appendChild(removeBtn);

            // Update the node's name in the global state (fcmState)
            const nodeId = parseInt(node.getAttribute('data-id'));
            const nodeInState = fcmState.nodes.find(n => n.id === nodeId);
            if (nodeInState) {
                nodeInState.name = cleanNodeName(newName);  // Clean and update in fcmState immediately
            }

            // Log the renaming in the timeline
            logChange(`Renamed node: ${cleanNodeName(oldName)} to ${cleanNodeName(newName)}`);  // Clean the log entry
            updateEdges(); // Ensure edges update with the correct node name
            updateNodeStats();
        }
    });
});

    // Attach hover event for showing the buttons
    node.addEventListener('mouseenter', () => {
        removeBtn.style.display = 'block';
        renameBtn.style.display = 'block';
    });

    node.addEventListener('mouseleave', () => {
        removeBtn.style.display = 'none';
        renameBtn.style.display = 'none';
    });

    // Initially hide the buttons
    removeBtn.style.display = 'none';
    renameBtn.style.display = 'none';

    // Mouse down event to start dragging
    node.addEventListener('mousedown', (e) => {
        isDragging = true;
        dragNode = node;
        offsetX = e.clientX - node.offsetLeft;
        offsetY = e.clientY - node.offsetTop;
        node.classList.add('dragging');
    });

    // Right-click (context menu) event to show the menu
    node.addEventListener('contextmenu', (e) => {
        e.preventDefault(); // Prevent the default right-click context menu
        contextNode = node; // Set the context node to the clicked node
        showContextMenu(e); // Show the custom context menu at the right position
    });
}




document.addEventListener('mousemove', (e) => {
if (isDragging && dragNode) {
    const mouseX = e.clientX - container.offsetLeft;
    const mouseY = e.clientY - container.offsetTop;

    // Ensure the node stays within the bounds of the container
    const newX = Math.min(Math.max(0, mouseX - offsetX), container.offsetWidth - dragNode.offsetWidth);
    const newY = Math.min(Math.max(0, mouseY - offsetY), container.offsetHeight - dragNode.offsetHeight);

    // Move node and update edges
    requestAnimationFrame(() => {
        dragNode.style.left = `${newX}px`;
        dragNode.style.top = `${newY}px`;
        updateEdges(); // Redraw edges in real-time while dragging
    });

    // Update the relative position in fcmState
    const nodeId = parseInt(dragNode.getAttribute('data-id'));
    const nodeInState = fcmState.nodes.find(node => node.id === nodeId);
    if (nodeInState) {
        nodeInState.x = newX / container.offsetWidth;
        nodeInState.y = newY / container.offsetHeight;
    }
}
});

function updateEdges() {
// Clear the canvas to redraw the edges
ctx.clearRect(0, 0, canvas.width, canvas.height);  

// Remove any existing "X" buttons before redrawing
const existingEdgeButtons = document.querySelectorAll('.edge-remove-btn');
existingEdgeButtons.forEach(btn => btn.remove());

edges.forEach((edge, index) => {
    const fromNode = edge.from.element;
    const toNode = edge.to.element;

    const fromX = fromNode.offsetLeft + fromNode.offsetWidth / 2;
    const fromY = fromNode.offsetTop + fromNode.offsetHeight / 2;
    const toX = toNode.offsetLeft + toNode.offsetWidth / 2;
    const toY = toNode.offsetTop + toNode.offsetHeight / 2;

    // Draw the edge with continuous arrows
    drawContinuousArrows(ctx, fromX, fromY, toX, toY, edge.value, edge.color);

    // Calculate the midpoint of the edge
    const midX = (fromX + toX) / 2;
    const midY = (fromY + toY) / 2;

    // Create a small "X" button at the midpoint
    const removeBtn = document.createElement('button');
    removeBtn.classList.add('edge-remove-btn');
    removeBtn.innerText = 'X';
    removeBtn.style.left = `${midX - 10}px`;  // Position the button at the midpoint
    removeBtn.style.top = `${midY - 10}px`;
    removeBtn.style.position = 'absolute';
    removeBtn.style.width = '20px';  // Small button size
    removeBtn.style.height = '20px';
    removeBtn.style.fontSize = '12px';
    removeBtn.style.cursor = 'pointer';
    removeBtn.style.backgroundColor = 'red';  // Red "X" button for visibility
    removeBtn.style.borderRadius = '50%';  // Make it circular
    removeBtn.style.border = 'none';
    removeBtn.style.color = 'white';

    // Add click event to remove the edge
    removeBtn.addEventListener('click', () => {
        edges.splice(index, 1);  // Remove the clicked edge from the edges array
        updateEdges();  // Redraw the edges after removal
        updateNodeStats();
    });

    // Append the "X" button to the container
    container.appendChild(removeBtn);
});
}

let selectedNode = null;

function onMouseMove(e) {
    if (isDragging && dragNode) {
        const containerRect = container.getBoundingClientRect();
        const mouseX = e.clientX - containerRect.left;
        const mouseY = e.clientY - containerRect.top;

        // Ensure the node stays within the bounds of the container
        const newX = Math.min(Math.max(0, mouseX - offsetX), container.offsetWidth - dragNode.offsetWidth);
        const newY = Math.min(Math.max(0, mouseY - offsetY), container.offsetHeight - dragNode.offsetHeight);

        // Move the node
        requestAnimationFrame(() => {
            dragNode.style.left = `${newX}px`;
            dragNode.style.top = `${newY}px`;
            updateEdges();  // Redraw edges in real-time while dragging
        });

        // Update the node's relative position in the node state
        const nodeId = parseInt(dragNode.getAttribute('data-id'));
        const nodeInState = nodes.find(node => node.id === nodeId);
        if (nodeInState) {
            nodeInState.x = newX / container.offsetWidth;  // Store relative position
            nodeInState.y = newY / container.offsetHeight;  // Store relative position
        }
    }
}



// Handle mouse up event to stop dragging
// Handle mouse up event to stop dragging
function onMouseUp() {
    isDragging = false;
    if (dragNode) {
        dragNode.classList.remove('dragging');
        dragNode = null;
    }
}


function onMouseDown(e) {
    const targetNode = e.target.closest('.node');  // Get the node element, even if the text is clicked
    if (targetNode) {
        isDragging = true;
        dragNode = targetNode;

        // Calculate the offset relative to the node itself, not the container
        offsetX = e.clientX - targetNode.getBoundingClientRect().left;
        offsetY = e.clientY - targetNode.getBoundingClientRect().top;

        targetNode.classList.add('dragging');
    }
}


// Helper function to get the node under the mouse (simplified)
function getNodeUnderMouse(x, y) {
// Example of how you might check if the mouse is over a node.
// This logic will depend on how you're storing and rendering your nodes.
return nodes.find(node => {
    return x >= node.x && x <= node.x + node.width &&
            y >= node.y && y <= node.y + node.height;
});
}

// Attach event listeners for dragging
document.addEventListener('mousedown', onMouseDown);
document.addEventListener('mousemove', onMouseMove);
document.addEventListener('mouseup', onMouseUp);

async function exportSnapshot() {
    if (selectedCheckpointId === null) {
        alert('Please select a checkpoint to export.');
        return;
    }

    if (!sessionPassword) {
        alert('Please set a session password to encrypt the export.');
        return;
    }

    const checkpoint = checkpoints.find(cp => cp.id === selectedCheckpointId);  // Use selectedCheckpointId
    if (!checkpoint) {
        alert('Selected checkpoint not found.');
        return;
    }

    const snapshotData = JSON.stringify(checkpoint.state);

    try {
        const encryptedBlob = await encryptData(sessionPassword, snapshotData);

        const link = document.createElement('a');
        link.href = URL.createObjectURL(encryptedBlob);
        link.download = `snapshot_${checkpoint.name.replace(/\s+/g, '_')}.snpfcmvss`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        console.log('Snapshot exported and encrypted successfully.');
    } catch (error) {
        console.error('Encryption failed:', error);
        alert('Failed to export the snapshot.');
    }
}


// Function to show the password modal popup
function showPasswordPopup(callback) {
    const modal = document.getElementById('password-popup-modal');
    const closeButton = document.querySelector('.password-popup-close');
    const submitButton = document.getElementById('password-popup-submit-btn');
    const passwordInput = document.getElementById('password-popup-input');

    modal.style.display = 'block'; // Show the modal

    // Close the modal when the close button is clicked
    closeButton.onclick = function () {
        modal.style.display = 'none';
        callback(null); // No password entered
    };

    // Close the modal when the user clicks outside the modal content
    window.onclick = function (event) {
        if (event.target == modal) {
            modal.style.display = 'none';
            callback(null); // No password entered
        }
    };

    // Handle password submission
    submitButton.onclick = function () {
        const password = passwordInput.value;
        modal.style.display = 'none'; // Hide the modal
        passwordInput.value = ''; // Clear the input
        callback(password); // Pass the entered password to the callback
    };
}

async function importSnapshot(files) {
    if (files.length === 0) {
        alert('No file selected.');
        return;
    }

    const file = files[0];
    const reader = new FileReader();

    reader.onload = async function (event) {
        try {
            const encryptedBlob = new Blob([event.target.result]);

            // Step 1: Try decryption with session password
            if (sessionPassword) {
                try {
                    const decryptedData = await decryptData(sessionPassword, encryptedBlob);
                    loadSnapshotData(JSON.parse(decryptedData));  // Proceed if decryption with session password succeeds
                    return;  // Exit if session password works
                } catch (error) {
                    console.warn('Session password decryption failed.');
                }
            }

            // Step 2: If session password fails, show the password popup to ask user for password
            showPasswordPopup(async function (manualPassword) {
                if (!manualPassword || manualPassword.trim() === '') {
                    alert('You must enter a valid password.');
                    return;
                }

                try {
                    const decryptedData = await decryptData(manualPassword, encryptedBlob);
                    loadSnapshotData(JSON.parse(decryptedData));  // Proceed if decryption with manual password succeeds
                } catch (error) {
                    console.error('Manual password decryption failed.');
                    alert('Failed to import the snapshot. Please ensure the password is correct and the file is valid.');
                }
            });

        } catch (error) {
            console.error('Error during import:', error);
            alert('Failed to import the snapshot.');
        }
    };

    reader.readAsArrayBuffer(file);
}

// Helper function to load snapshot data
function loadSnapshotData(snapshotData) {
    if (!snapshotData.nodes || !snapshotData.edges) {
        alert('Invalid snapshot file.');
        return;
    }

    alert('Snapshot loaded successfully.');
    fcmState.nodes = snapshotData.nodes;
    fcmState.edges = snapshotData.edges;
    clearFCM();
    redrawFCM();

    // Update the node counter to prevent ID conflicts
    const highestNodeId = Math.max(...snapshotData.nodes.map(node => node.id));
    nodeCounter = highestNodeId + 1;

    // Automatically create a checkpoint for the imported snapshot
    saveCheckpoint(`Imported Snapshot - ${new Date().toLocaleString()}`);
}


function highlightCheckpointButton(id) {
    // Remove the highlight from all checkpoint buttons
    const buttons = document.querySelectorAll('#checkpoint-tray button');
    buttons.forEach(button => {
        button.classList.remove('selected-checkpoint');
    });

    // Add the highlight to the clicked button and set selectedCheckpointId
    const selectedButton = document.querySelector(`#checkpoint-tray button[data-id='${id}']`);
    if (selectedButton) {
        selectedButton.classList.add('selected-checkpoint');
        selectedCheckpointId = id;  // Ensure selectedCheckpointId is always updated
    } else {
        console.error(`Checkpoint with id ${id} not found.`);
    }
}

function addNode() {
    // If nodes already exist, set nodeCounter to the highest node number + 1
    if (nodes.length > 0) {
        const highestNodeId = Math.max(...nodes.map(node => node.id));
        nodeCounter = highestNodeId + 1;
    } else {
        nodeCounter = 1;  // If no nodes exist, start from 1
    }

    const node = document.createElement('div');
    node.classList.add('node');
    const nodeName = `Node ${nodeCounter}`;  // Use the updated nodeCounter for the new node ID
    node.innerText = cleanNodeName(nodeName);  // Ensure the node name is clean

    // Set random positions and ensure they are within the container's bounds
    const leftPos = Math.random() * (container.offsetWidth - 60);
    const topPos = Math.random() * (container.offsetHeight - 60);

    node.style.left = `${leftPos}px`;
    node.style.top = `${topPos}px`;
    node.setAttribute('data-id', nodeCounter);  // Assign the current nodeCounter as the node's ID

    // Create the "X" (Remove) button
    const removeBtn = document.createElement('button');
    removeBtn.classList.add('node-remove-btn');
    removeBtn.innerText = 'X';
    removeBtn.style.position = 'absolute';
    removeBtn.style.right = '0px';  // Align on the right of the node
    removeBtn.style.top = '0px';    // Align at the top of the node

    // Create the "R" (Rename) button
    const renameBtn = document.createElement('button');
    renameBtn.classList.add('node-rename-btn');
    renameBtn.innerText = 'R';
    renameBtn.style.position = 'absolute';
    renameBtn.style.left = '0px';   // Align on the left of the node
    renameBtn.style.top = '0px';    // Align at the top of the node

    // Append buttons to the node
    node.appendChild(removeBtn);
    node.appendChild(renameBtn);

    container.appendChild(node);
    nodes.push({ id: nodeCounter, element: node });

    // Store node in fcmState with relative positions (percentage-based)
    fcmState.nodes.push({
        id: nodeCounter,
        x: leftPos / container.offsetWidth,  // Relative to container width
        y: topPos / container.offsetHeight,  // Relative to container height
        name: nodeName
    });

    // Event listener to remove node
    removeBtn.addEventListener('click', () => {
        const nodeId = parseInt(node.getAttribute('data-id')); // Get the node's ID

        // Remove the node element from the DOM
        container.removeChild(node);

        // Remove the node from the nodes array
        nodes = nodes.filter(n => n.id !== nodeId);

        // Remove edges that are connected to the node
        edges = edges.filter(edge => edge.from.id !== nodeId && edge.to.id !== nodeId);

        // Update the edges on the canvas after removing the node and its edges
        updateEdges();
        
        logChange(`Deleted node: ${cleanNodeName(nodeName)}`);
    });

    // Event listener to rename node
    renameBtn.addEventListener('click', () => {
        showModal("Enter the new name for this node:", (newName) => {
            if (newName !== null && newName.trim() !== "") {
                const oldName = node.innerText; // Store the old name for logging purposes

                // Remove old text and replace it with the new name
                while (node.firstChild) {
                    node.removeChild(node.firstChild); // Clear existing text and buttons
                }

                const nodeText = document.createElement('span');
                nodeText.innerText = newName;
                node.appendChild(nodeText);
                node.appendChild(renameBtn);
                node.appendChild(removeBtn);

                // Update the node's name in the global state
                const nodeId = parseInt(node.getAttribute('data-id'));
                const nodeInState = fcmState.nodes.find(n => n.id === nodeId);
                if (nodeInState) {
                    nodeInState.name = newName;  // Update in fcmState immediately
                }

                // Log the renaming in the timeline
                logChange(`Renamed node: ${cleanNodeName(oldName)} to ${cleanNodeName(newName)}`);

                // Update any other logic or UI components that might reference this node's name
                updateEdges(); // Ensure edges update with the correct node name
            }
        });
    });

    attachNodeEventListeners(node);  // Ensure the node becomes draggable
    logChange(`Added node: ${cleanNodeName(nodeName)}`);
    nodeCounter++;  // Increment nodeCounter after adding the node
    updateNodeStats();

}

// Function to generate a crypto key from a password using PBKDF2
async function deriveKey(password, salt, iterations = 100000) {
    const encoder = new TextEncoder();
    const keyMaterial = await crypto.subtle.importKey(
        "raw",
        encoder.encode(password),
        { name: "PBKDF2" },
        false,
        ["deriveKey"]
    );

    return await crypto.subtle.deriveKey(
        {
            name: "PBKDF2",
            salt: encoder.encode(salt),
            iterations: iterations,
            hash: "SHA-256"
        },
        keyMaterial,
        { name: "AES-GCM", length: 256 },
        false,
        ["encrypt", "decrypt"]
    );
}

// Encrypt function using AES-GCM
async function encryptData(password, data) {
    const salt = crypto.getRandomValues(new Uint8Array(16));
    const iv = crypto.getRandomValues(new Uint8Array(12)); // Initialization vector for AES-GCM
    const key = await deriveKey(password, salt);

    const encoder = new TextEncoder();
    const encrypted = await crypto.subtle.encrypt(
        {
            name: "AES-GCM",
            iv: iv
        },
        key,
        encoder.encode(data) // Data to encrypt
    );

    // Concatenate salt, IV, and encrypted data
    const encryptedBlob = new Blob([salt, iv, new Uint8Array(encrypted)]);
    return encryptedBlob;
}

// Decrypt function using AES-GCM
async function decryptData(password, encryptedBlob) {
    const arrayBuffer = await encryptedBlob.arrayBuffer();
    const salt = new Uint8Array(arrayBuffer.slice(0, 16));
    const iv = new Uint8Array(arrayBuffer.slice(16, 28));
    const encryptedData = arrayBuffer.slice(28);

    const key = await deriveKey(password, salt);

    try {
        const decrypted = await crypto.subtle.decrypt(
            {
                name: "AES-GCM",
                iv: iv
            },
            key,
            encryptedData
        );

        const decoder = new TextDecoder();
        return decoder.decode(decrypted); // Return decrypted data as string
    } catch (err) {
        throw new Error("Invalid password or corrupted data.");
    }
}

// Function to simulate a button click
function triggerButton(buttonId) {
    const button = document.getElementById(buttonId);
    if (button) {
        button.click(); // Simulate the click event
    }
}

// Add event listener for keydown to handle shortcuts
document.addEventListener('keydown', function (event) {
    // Example: Ctrl + S for "Save Project"
    if (event.altKey && event.key === 's') {
        event.preventDefault(); // Prevent the default browser save behavior
        saveProject()
    }

    // Example: Ctrl + L for "Load Project"
    if (event.altKey && event.key === 'o') {
        event.preventDefault();
        loadProject()
    }

    // Example: Ctrl + N for "Add Node"
    if (event.altKey && event.key === 'n') {
        event.preventDefault();
        addNode()
    }

    // Example: Ctrl + E for "Add Edge"
    if (event.altKey && event.key === 'e') {
        event.preventDefault();
        addEdge()
    }
});


document.addEventListener('DOMContentLoaded', function() {
    const overlay = document.getElementById('password-overlay');
    const enterPass = document.getElementById('password-input');
    const passwordSubmitBtn = document.getElementById('password-submit-btn');

    passwordSubmitBtn.addEventListener('click', function() {
        const password = enterPass.value.trim(); // Get the value entered by the user

        if (password) {
            sessionPassword = password;  // Set the sessionPassword to the entered value
            overlay.style.display = 'none';  // Hide the password modal after setting the password
            alert('Session password set successfully!');
        } else {
            alert('Please enter a valid password.');  // Prompt user if the password is empty
        }
    });

    // Prevent bypassing the overlay by blocking access to the rest of the page
    window.addEventListener('beforeunload', function(e) {
        if (overlay.style.display !== 'none') {
            e.preventDefault();
        }
    });
});

function logChange(action) {
    const timelineTray = document.getElementById('timeline-tray');
    const timestamp = new Date().toLocaleTimeString();  // Get current time
    const logEntry = document.createElement('p');
    logEntry.innerText = `${timestamp}: ${cleanNodeName(action)}`;  // Ensure all logged actions are cleaned
    timelineTray.appendChild(logEntry);
    
    // Scroll to the bottom so the latest action is visible
    timelineTray.scrollTop = timelineTray.scrollHeight;
}
// Function to calculate node type based on indegree and outdegree
function getNodeType(indegree, outdegree) {
    if (indegree === 0 && outdegree > 0) {
        return 'Driver'; // No incoming connections but has outgoing connections
    } else if (outdegree === 0 && indegree > 0) {
        return 'Receiver'; // No outgoing connections but has incoming connections
    } else if (indegree == 0 && outdegree == 0) {
        return 'N/A'; // Both incoming and outgoing connections
    } else {
        return 'Intermediate'; // No connections at all
    }
}
function updateNodeStats() {
    const tableBody = document.querySelector('#node-stats-table tbody');
    const weightMatrixHead = document.querySelector('#weight-matrix-table thead tr');
    const weightMatrixBody = document.querySelector('#weight-matrix-table tbody');

    // Clear the tables before adding new rows
    tableBody.innerHTML = ''; 
    weightMatrixHead.innerHTML = '<th>From / To</th>'; 
    weightMatrixBody.innerHTML = '';

    // Get node IDs and names to create the tables
    const nodeIds = nodes.map(node => node.id);
    const nodeNames = nodes.map(node => node.element.innerText);

    // Populate the Weight Matrix header with node names
    nodeNames.forEach(name => {
        const th = document.createElement('th');
        th.textContent = name;
        weightMatrixHead.appendChild(th);
    });

    // Iterate through nodes to populate the Node Stats table and Weight Matrix
    nodes.forEach((node, nodeIndex) => {
        let indegree = 0;
        let outdegree = 0;
        let indegreeValueSum = 0;
        let outdegreeValueSum = 0;
        let centralityValueSum = 0;

        // Calculate indegree, outdegree, and centrality for the current node
        edges.forEach(edge => {
            if (edge.to.id === node.id) {
                indegree++;
                indegreeValueSum += parseFloat(edge.value);
            }
            if (edge.from.id === node.id) {
                outdegree++;
                outdegreeValueSum += parseFloat(edge.value);
            }
        });

        // Centrality is the sum of indegree and outdegree
        centralityValueSum = indegreeValueSum + outdegreeValueSum;

        // Create a new row for the Node Stats table
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${cleanNodeName(node.element.innerText)}</td>
            <td>${indegree}</td>
            <td>${indegreeValueSum.toFixed(3)}</td>
            <td>${outdegree}</td>
            <td>${outdegreeValueSum.toFixed(3)}</td>
            <td>${indegree + outdegree}</td>
            <td>${centralityValueSum.toFixed(3)}</td>
            <td>${getNodeType(indegreeValueSum, outdegreeValueSum)}</td>
        `;
        tableBody.appendChild(row);

        // Now populate the Weight Matrix row for the current node
        const matrixRow = document.createElement('tr');
        const matrixRowHeader = document.createElement('th');
        matrixRowHeader.textContent = cleanNodeName(node.element.innerText);
        matrixRow.appendChild(matrixRowHeader);

        // Iterate over each node and populate the matrix cell with the corresponding weight
        nodeIds.forEach(targetNodeId => {
            const cell = document.createElement('td');
            const edge = edges.find(edge => edge.from.id === node.id && edge.to.id === targetNodeId);

            if (edge) {
                cell.textContent = edge.value;
                applyColor(cell, parseFloat(edge.value)); // Colorize the cell based on value
            } else {
                cell.textContent = '-'; // No connection between nodes
            }

            matrixRow.appendChild(cell);
        });

        weightMatrixBody.appendChild(matrixRow);
    });
}

// Apply color function to colorize table cells based on value
function applyColor(cell, value) {
    if (value > 0) {
        cell.style.backgroundColor = 'lightgreen'; // Green for positive values
    } else if (value < 0) {
        cell.style.backgroundColor = 'lightcoral'; // Red for negative values
    } else {
        cell.style.backgroundColor = 'white'; // White for zero
    }
}
    function downloadTableAsCSV(tableId) {
        const table = document.getElementById(tableId);
        let csvContent = "";

        for (let row of table.rows) {
            let rowData = [];
            for (let cell of row.cells) {
                rowData.push(cell.innerText);
            }
            csvContent += rowData.join(",") + "\n";
        }

        const blob = new Blob([csvContent], { type: 'text/csv' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = `${tableId}.csv`;
        link.click();
    }


// Function to get the weight matrix and node names from the HTML table
function getWeightMatrix() {
    let table = document.getElementById('weight-matrix-table');
    let matrix = [];
    let nodeNames = [];

    // Get the node names from the header row (first row, skip the first cell)
    const headerCells = table.rows[0].cells;
    for (let i = 1; i < headerCells.length; i++) {
        nodeNames.push(headerCells[i].innerText);  // Extract node name
    }

    // Get the weight matrix values (skip the first column and row headers)
    for (let i = 1; i < table.rows.length; i++) {
        let row = [];
        for (let j = 1; j < table.rows[i].cells.length; j++) {
            let cellValue = parseFloat(table.rows[i].cells[j].innerText);
            row.push(isNaN(cellValue) ? 0 : cellValue);
        }
        matrix.push(row);
    }

    return { matrix, nodeNames };  // Return both the matrix and the node names
}

function kosko() {
    console.log('Initiating Kosko evaluation...');

    // Get the weight matrix and node names
    const { matrix: weightMatrix, nodeNames } = getWeightMatrix();
    console.log('Weight Matrix:', weightMatrix);
    console.log('Node Names:', nodeNames);

    // Prevent clearing of input fields by skipping re-population if already populated
    const container = document.getElementById('node-fields-container');
    if (container.innerHTML === '') {
        populateNodeFields(nodeNames, document.getElementById('activation-value').value);
    }

    // Get all node fields and identify nodes with non-zero tweak values
    const nodeFields = document.querySelectorAll('.node-field');
    let fixedNodes = [];  // This will store the node names of tweaked nodes
    let fixedNodeValues = {};  // This will store the tweak value for each fixed node

    // Read tweak inputs and activation inputs (if applicable)
    nodeFields.forEach((nodeField, index) => {
        const tweakInput = nodeField.querySelector('input[placeholder="Tweak"]');
        const tweakValue = parseFloat(tweakInput.value);

        if (tweakValue !== 0) {
            const nodeName = nodeNames[index];
            fixedNodes.push(nodeName);
            fixedNodeValues[nodeName] = tweakValue;  // Store the tweak value for this node
        }
    });

    console.log('Fixed Nodes:', fixedNodes);
    console.log('Fixed Node Values:', fixedNodeValues);

    // Allow the user to choose the transfer function, activation type, and inference mechanism
    const transferFunctionType = document.getElementById('transfer-function').value;
    const activationType = document.getElementById('activation-value').value;
    const inferenceMechanismType = document.getElementById('inference-mechanism').value;

    let iterations = 100;
    let convergenceThreshold = 0.005;  // Threshold to check for convergence

    // Define the transfer functions
    function sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    function tanh(x) {
        return Math.tanh(x);
    }

    function bivalent(x) {
        return x >= 0 ? 1 : -1;
    }

    function trivalent(x) {
        if (x > 0.5) return 1;
        if (x < -0.5) return -1;
        return 0;
    }

    // Select the transfer function based on the user's choice
    let transferFunction;
    switch (transferFunctionType) {
        case 'sigmoid':
            transferFunction = sigmoid;
            break;
        case 'tanh':
            transferFunction = tanh;
            break;
        case 'bivalent':
            transferFunction = bivalent;
            break;
        case 'trivalent':
            transferFunction = trivalent;
            break;
        default:
            transferFunction = sigmoid;  // Default to sigmoid if none selected
            console.warn("Invalid transfer function selected, defaulting to sigmoid.");
    }

    // Define activation value generators
    function randomActivation(numNodes) {
        return Array(numNodes).fill(0).map(() => Math.random());
    }

    function medianActivation(numNodes) {
        return Array(numNodes).fill(0.5);
    }

    function zeroActivation(numNodes) {
        return Array(numNodes).fill(0);
    }

    function fullActivation(numNodes) {
        return Array(numNodes).fill(1);
    }

    function manualActivation(numNodes) {
        const activationValues = [];
        for (let i = 0; i < numNodes; i++) {
            let value = parseFloat(document.getElementById(`manual-activation-${i}`).value);
            activationValues.push(isNaN(value) ? 0 : value);  // Default to 0 if no value is provided
        }
        return activationValues;
    }

    // Select the initial activation state based on the user's choice
    let initialState;
    const numNodes = weightMatrix[0].length;
    switch (activationType) {
        case 'random':
            initialState = randomActivation(numNodes);
            break;
        case 'median':
            initialState = medianActivation(numNodes);
            break;
        case 'zero':
            initialState = zeroActivation(numNodes);
            break;
        case 'full':
            initialState = fullActivation(numNodes);
            break;
        case 'manual':
            initialState = manualActivation(numNodes);
            break;
        default:
            initialState = medianActivation(numNodes);  // Default to median if none selected
            console.warn("Invalid activation type selected, defaulting to median activation.");
    }

    // Override the initialState for the fixed nodes with their tweak values
    fixedNodes.forEach(node => {
        const nodeIndex = nodeNames.indexOf(node);
        if (nodeIndex !== -1) {
            initialState[nodeIndex] = fixedNodeValues[node];  // Set to the fixed value
        }
    });

    let currentState = [...initialState];
    let results = [currentState.slice()];  // Add the initial state

    let convergenceStatus = false;
    let stepCount = 0;

    // Define the inference mechanisms
    function koskoInference(state, weightMatrix) {
        let result = [];
        for (let i = 0; i < weightMatrix[0].length; i++) {
            let sum = 0;
            for (let j = 0; j < weightMatrix.length; j++) {
                sum += weightMatrix[j][i] * state[j];
            }
            result.push(sum + state[i]);  
        }
        return result;
    }

    function modKoskoInference(state, weightMatrix) {
        let result = [];
        for (let i = 0; i < weightMatrix[0].length; i++) {
            let sum = 0;
            for (let j = 0; j < weightMatrix.length; j++) {
                sum += weightMatrix[j][i] * state[j];
            }
            result.push(sum);
        }
        return result;
    }

    function rescaledInference(state, weightMatrix) {
        let rescaledState = state.map(x => 2 * x - 1);  // Rescale to [-1, 1]
        let result = [];
        for (let i = 0; i < weightMatrix[0].length; i++) {
            let sum = 0;
            for (let j = 0; j < weightMatrix.length; j++) {
                sum += weightMatrix[j][i] * rescaledState[j];
            }
            result.push(sum + rescaledState[i]);
        }
        return result.map(x => (x + 1) / 2);  // Rescale back to [0, 1]
    }

    // Select the inference mechanism based on the user's choice
    let inferenceFunction;
    switch (inferenceMechanismType) {
        case 'kosko':
            inferenceFunction = koskoInference;
            break;
        case 'mkosko':
            inferenceFunction = modKoskoInference;
            break;
        case 'rescaled':
            inferenceFunction = rescaledInference;
            break;
        default:
            inferenceFunction = koskoInference;  // Default to Kosko if none selected
            console.warn("Invalid inference mechanism selected, defaulting to Kosko.");
    }

    // Check for convergence between two consecutive states
    function checkConvergence(results) {
        const lastResult = results[results.length - 1];
        const secondLastResult = results[results.length - 2];

        let maxDiff = 0;
        for (let i = 0; i < lastResult.length; i++) {
            const diff = Math.abs(lastResult[i] - secondLastResult[i]);
            if (diff > maxDiff) {
                maxDiff = diff;
            }
        }
        return maxDiff <= convergenceThreshold;
    }

    // Main simulation loop
    for (let i = 0; i < iterations; i++) {
        if (!convergenceStatus) {
            // Perform inference using the selected inference mechanism
            let inferredState = inferenceFunction(currentState, weightMatrix);
            currentState = inferredState.map(transferFunction);  // Apply the chosen transfer function

            // Fix activation values of nodes with non-zero tweak values
            fixedNodes.forEach(node => {
                const nodeIndex = nodeNames.indexOf(node);
                if (nodeIndex !== -1) {
                    // Preserve the activation value to the tweak value
                    currentState[nodeIndex] = fixedNodeValues[node];
                }
            });

            // Record the current state
            results.push(currentState.slice());
            console.log(currentState.slice());

            // Increment step count
            stepCount++;

            // Check for convergence
            convergenceStatus = checkConvergence(results);
        } else {
            console.log(`The values converged in the ${stepCount + 1} state (residual <= ${convergenceThreshold})`);
            break;
        }

        // If the maximum number of iterations is reached without convergence
        if (stepCount >= iterations) {
            console.warn("The values didn't converge. More iterations are required!");
            break;
        }
    }

    // Update the results table and line plot with the node names
    updateKoskoResultsTable(results, nodeNames);
    generateLinePlot(results, nodeNames);
    // Keep node fields populated with existing values
    populateNodeFields(nodeNames, activationType);
    showKoskoLineChart();
    showKoskoResultsTable();
}

function updateKoskoResultsTable(results, nodeNames) {
    const table = document.getElementById('kosko-results-table');
    const tableHead = table.querySelector('thead tr');
    const tableBody = table.querySelector('tbody');

    // Clear the existing table content before populating new results
    tableHead.innerHTML = '<th>Iteration</th>';
    tableBody.innerHTML = '';

    // Populate the header with node names
    nodeNames.forEach(nodeName => {
        const th = document.createElement('th');
        th.textContent = nodeName;
        tableHead.appendChild(th);
    });

    // Populate the table body with results for each iteration
    results.forEach((state, iteration) => {
        const row = document.createElement('tr');
        const iterationCell = document.createElement('td');
        iterationCell.textContent = iteration;
        row.appendChild(iterationCell);

        state.forEach(value => {
            const cell = document.createElement('td');
            cell.textContent = value.toFixed(5); // Show values with 5 decimal places
            row.appendChild(cell);
        });

        tableBody.appendChild(row);
    });
}


let koskoChart; // Declare chart variable globally

function generateLinePlot(results, nodeNames) {
    console.log('Generating Line Plot with Results:', results); // Debugging log
    const labels = results.map((_, index) => `Iteration ${index}`);

    const datasets = [];
    const nodeCount = results[0].length;
    for (let i = 0; i < nodeCount; i++) {
        const nodeData = results.map(state => state[i]);
        datasets.push({
            label: nodeNames[i],  // Use node name instead of "Node {i+1}"
            data: nodeData,
            fill: false,  // Disable fill to make lines more visible
            borderColor: getRandomColor(),
            backgroundColor: function (context) {  // Add gradient fill
                const chart = context.chart;
                const {ctx, chartArea} = chart;

                if (!chartArea) {
                    return null;
                }
                return createSubtleGradient(ctx, chartArea);  // Apply a more subtle gradient
            },
            borderWidth: 3,
            pointRadius: 5,
            pointHoverRadius: 7,
            pointBackgroundColor: '#ffffff',
            pointBorderColor: getRandomColor(),
            pointBorderWidth: 2,
            tension: 0.4  // Smooth the lines more
        });
    }

    const ctx = document.getElementById('kosko-line-chart').getContext('2d');
    if (koskoChart) {
        console.log('Destroying previous chart'); // Debugging log
        koskoChart.destroy();  // Destroy existing chart if it exists
    }

    console.log('Creating new chart');  // Debugging log
    koskoChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Kosko Simulation Results',
                    font: {
                        size: 24,
                        family: 'Arial, sans-serif',
                        weight: 'bold'
                    },
                    padding: {
                        top: 20,
                        bottom: 30
                    },
                    color: '#333'
                },
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        font: {
                            size: 14,
                            family: 'Arial, sans-serif'
                        },
                        color: '#666'
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(0,0,0,0.7)',
                    titleFont: {
                        size: 16,
                        weight: 'bold'
                    },
                    bodyFont: {
                        size: 14
                    },
                    borderColor: '#ddd',
                    borderWidth: 1,
                    callbacks: {
                        label: function (tooltipItem) {
                            return `${tooltipItem.dataset.label}: ${tooltipItem.raw.toFixed(2)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    ticks: {
                        font: {
                            size: 12
                        },
                        color: '#555'
                    }
                },
                y: {
                    beginAtZero: true,
                    suggestedMax: 1,
                    grid: {
                        color: 'rgba(200, 200, 200, 0.3)'
                    },
                    ticks: {
                        font: {
                            size: 12
                        },
                        color: '#555'
                    }
                }
            }
        }
    });
}

function getRandomColor() {
    const letters = '0123456789ABCDEF';
    let color = '#';
    for (let i = 0; i < 6; i++) { color += letters[Math.floor(Math.random() * 16)]; }
    return color;
}

// Create a more subtle gradient with lighter opacity
function createSubtleGradient(ctx, chartArea) {
    const gradient = ctx.createLinearGradient(0, chartArea.bottom, 0, chartArea.top);
    gradient.addColorStop(0, 'rgba(58, 123, 213, 0.1)');  // Lighter opacity for blue
    gradient.addColorStop(0.5, 'rgba(0, 210, 255, 0.1)');  // Lighter cyan
    gradient.addColorStop(1, 'rgba(0, 255, 133, 0.1)');  // Lighter green
    return gradient;
}

document.getElementById('activation-value').addEventListener('change', function () {
    const activationType = this.value;
    const { matrix: weightMatrix, nodeNames } = getWeightMatrix();
    populateNodeFields(nodeNames, activationType);  // Pass activationType to populateNodeFields
});
function populateNodeFields(nodeNames, activationType) {
    const container = document.getElementById('node-fields-container');
    container.innerHTML = '';  // Clear any existing content

    nodeNames.forEach((nodeName, index) => {  // Add index to uniquely identify inputs
        // Create a div for each node field
        const nodeField = document.createElement('div');
        nodeField.classList.add('node-field');

        // Create the header for the node
        const nodeHeader = document.createElement('div');
        nodeHeader.classList.add('node-header');
        nodeHeader.textContent = nodeName;

        // Create input for tweaking value
        const tweakInput = document.createElement('input');
        tweakInput.type = 'number';
        tweakInput.placeholder = 'Tweak';
        tweakInput.step = '0.01';
        tweakInput.value = 0.0;  // Default to 0
        tweakInput.classList.add('node-input');
        tweakInput.id = `tweak-input-${index}`;  // Unique id for the tweak input
        tweakInput.style.marginLeft = '10px';  // Inline style for margin-left
        tweakInput.style.marginRight = '10px'; // Inline style for margin-right

        // Create label for tweak value
        const tweakLabel = document.createElement('label');
        tweakLabel.textContent = 'Tweak:';
        tweakLabel.classList.add('input-label');
        tweakLabel.style.marginLeft = '10px';  // Inline style for margin-left
        tweakLabel.style.marginRight = '10px'; // Inline style for margin-right

        // Append the tweak input and label
        nodeField.appendChild(nodeHeader);
        nodeField.appendChild(tweakLabel);
        nodeField.appendChild(tweakInput);

        // Add activation input field if "Manual" is selected
        if (activationType === 'manual') {
            const activationInput = document.createElement('input');
            activationInput.type = 'number';
            activationInput.placeholder = 'Activation';
            activationInput.step = '0.01';
            activationInput.value = 0.5;  // Default to 0.5
            activationInput.classList.add('node-input');
            activationInput.id = `manual-activation-${index}`;  // Unique id for manual activation
            activationInput.style.marginLeft = '10px';  // Inline style for margin-left
            activationInput.style.marginRight = '10px'; // Inline style for margin-right

            const activationLabel = document.createElement('label');
            activationLabel.textContent = 'Activation:';
            activationLabel.classList.add('input-label');
            activationLabel.style.marginLeft = '10px';  // Inline style for margin-left
            activationLabel.style.marginRight = '10px'; // Inline style for margin-right

            // Append activation input and label
            nodeField.appendChild(activationLabel);
            nodeField.appendChild(activationInput);
        }

        // Append the node field to the container
        container.appendChild(nodeField);
    });
}



// Initial population based on the current activation value
window.onload = function () {
    const { matrix: weightMatrix, nodeNames } = getWeightMatrix();
    const activationType = document.getElementById('activation-value').value;
    populateNodeFields(nodeNames, activationType);  // Initial population
};

function fixation() {
    console.log('Initiating fixation evaluation with tweak values...');

    // Get the weight matrix and node names
    const { matrix: weightMatrix, nodeNames } = getWeightMatrix();
    console.log('Weight Matrix:', weightMatrix);
    console.log('Node Names:', nodeNames);

    // Get all node fields and identify nodes with non-zero tweak values
    const nodeFields = document.querySelectorAll('.node-field');
    let fixedNodes = [];  // This will store the node names of tweaked nodes
    let fixedNodeValues = {};  // This will store the tweak value for each fixed node

    // Read existing tweak inputs and activation inputs (if applicable)
    nodeFields.forEach((nodeField, index) => {
        const tweakInput = nodeField.querySelector('input[placeholder="Tweak"]');
        const tweakValue = parseFloat(tweakInput.value);

        if (tweakValue !== 0) {
            const nodeName = nodeNames[index];
            fixedNodes.push(nodeName);
            fixedNodeValues[nodeName] = tweakValue;  // Store the tweak value for this node
        }
    });

    console.log('Fixed Nodes:', fixedNodes);
    console.log('Fixed Node Values:', fixedNodeValues);

    // Allow the user to choose the transfer function, activation type, and inference mechanism
    const transferFunctionType = document.getElementById('transfer-function').value;
    const activationType = document.getElementById('activation-value').value;
    const inferenceMechanismType = document.getElementById('inference-mechanism').value;

    let iterations = 100;
    let convergenceThreshold = 0.005;  // Threshold to check for convergence

    // Define the transfer functions
    function sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    function tanh(x) {
        return Math.tanh(x);
    }

    function bivalent(x) {
        return x >= 0 ? 1 : -1;
    }

    function trivalent(x) {
        if (x > 0.5) return 1;
        if (x < -0.5) return -1;
        return 0;
    }

    // Select the transfer function based on the user's choice
    let transferFunction;
    switch (transferFunctionType) {
        case 'sigmoid':
            transferFunction = sigmoid;
            break;
        case 'tanh':
            transferFunction = tanh;
            break;
        case 'bivalent':
            transferFunction = bivalent;
            break;
        case 'trivalent':
            transferFunction = trivalent;
            break;
        default:
            transferFunction = sigmoid;  // Default to sigmoid if none selected
            console.warn("Invalid transfer function selected, defaulting to sigmoid.");
    }

    // Define activation value generators
    function randomActivation(numNodes) {
        return Array(numNodes).fill(0).map(() => Math.random());
    }

    function medianActivation(numNodes) {
        return Array(numNodes).fill(0.5);
    }

    function zeroActivation(numNodes) {
        return Array(numNodes).fill(0);
    }

    function fullActivation(numNodes) {
        return Array(numNodes).fill(1);
    }

    function manualActivation(numNodes) {
        const activationValues = [];
        for (let i = 0; i < numNodes; i++) {
            let value = parseFloat(document.getElementById(`manual-activation-${i}`).value);
            activationValues.push(isNaN(value) ? 0 : value);  // Default to 0 if no value is provided
        }
        return activationValues;
    }

    // Select the initial activation state based on the user's choice
    let initialState;
    const numNodes = weightMatrix[0].length;
    switch (activationType) {
        case 'random':
            initialState = randomActivation(numNodes);
            break;
        case 'median':
            initialState = medianActivation(numNodes);
            break;
        case 'zero':
            initialState = zeroActivation(numNodes);
            break;
        case 'full':
            initialState = fullActivation(numNodes);
            break;
        case 'manual':
            initialState = manualActivation(numNodes);
            break;
        default:
            initialState = medianActivation(numNodes);  // Default to median if none selected
            console.warn("Invalid activation type selected, defaulting to median activation.");
    }

    // Override the initialState for the fixed nodes with their tweak values
    fixedNodes.forEach(node => {
        const nodeIndex = nodeNames.indexOf(node);
        if (nodeIndex !== -1) {
            initialState[nodeIndex] = fixedNodeValues[node];  // Set to the fixed value
        }
    });

    let currentState = [...initialState];
    let results = [currentState.slice()];  // Add the initial state

    let convergenceStatus = false;
    let stepCount = 0;

    // Define the inference mechanisms
    function koskoInference(state, weightMatrix) {
        let result = [];
        for (let i = 0; i < weightMatrix[0].length; i++) {
            let sum = 0;
            for (let j = 0; j < weightMatrix.length; j++) {
                sum += weightMatrix[j][i] * state[j];
            }
            result.push(sum + state[i]);  
        }
        return result;
    }

    function modKoskoInference(state, weightMatrix) {
        let result = [];
        for (let i = 0; i < weightMatrix[0].length; i++) {
            let sum = 0;
            for (let j = 0; j < weightMatrix.length; j++) {
                sum += weightMatrix[j][i] * state[j];
            }
            result.push(sum); 
        }
        return result;
    }

    function rescaledInference(state, weightMatrix) {
        let rescaledState = state.map(x => 2 * x - 1);  // Rescale to [-1, 1]
        let result = [];
        for (let i = 0; i < weightMatrix[0].length; i++) {
            let sum = 0;
            for (let j = 0; j < weightMatrix.length; j++) {
                sum += weightMatrix[j][i] * rescaledState[j];
            }
            result.push(sum + rescaledState[i]);
        }
        return result.map(x => (x + 1) / 2);  // Rescale back to [0, 1]
    }

    // Select the inference mechanism based on the user's choice
    let inferenceFunction;
    switch (inferenceMechanismType) {
        case 'kosko':
            inferenceFunction = koskoInference;
            break;
        case 'mkosko':
            inferenceFunction = modKoskoInference;
            break;
        case 'rescaled':
            inferenceFunction = rescaledInference;
            break;
        default:
            inferenceFunction = koskoInference;  // Default to Kosko if none selected
            console.warn("Invalid inference mechanism selected, defaulting to Kosko.");
    }

    // Check for convergence between two consecutive states
    function checkConvergence(results) {
        const lastResult = results[results.length - 1];
        const secondLastResult = results[results.length - 2];

        let maxDiff = 0;
        for (let i = 0; i < lastResult.length; i++) {
            const diff = Math.abs(lastResult[i] - secondLastResult[i]);
            if (diff > maxDiff) {
                maxDiff = diff;
            }
        }
        return maxDiff <= convergenceThreshold;
    }

    // Main simulation loop
    for (let i = 0; i < iterations; i++) {
        if (!convergenceStatus) {
            // Perform inference using the selected inference mechanism
            let inferredState = inferenceFunction(currentState, weightMatrix);
            currentState = inferredState.map(transferFunction);  // Apply the chosen transfer function

            // Fix activation values of nodes with non-zero tweak values
            fixedNodes.forEach(node => {
                const nodeIndex = nodeNames.indexOf(node);
                if (nodeIndex !== -1) {
                    // Preserve the activation value to the tweak value
                    currentState[nodeIndex] = fixedNodeValues[node];
                }
            });

            // Record the current state
            results.push(currentState.slice());
            console.log(currentState.slice());

            // Increment step count
            stepCount++;

            // Check for convergence
            convergenceStatus = checkConvergence(results);
        } else {
            console.log(`The values converged in the ${stepCount + 1} state (residual <= ${convergenceThreshold})`);
            break;
        }

        // If the maximum number of iterations is reached without convergence
        if (stepCount >= iterations) {
            console.warn("The values didn't converge. More iterations are required!");
            break;
        }
    }
    // Update the results table and line plot with the node names
    updateFixationResultsTable(results, nodeNames);
}

// Declare a global chart variable to handle updating/destroying the chart
let differenceBarChart;

function updateFixationResultsTable(results, nodeNames) {
    const table = document.getElementById('fixation-results-table');
    const tableHead = table.querySelector('thead tr');
    const tableBody = table.querySelector('tbody');

    // Clear the existing table content before populating new results
    tableHead.innerHTML = '<th>Iteration</th>';
    tableBody.innerHTML = '';

    // Populate the header with node names
    nodeNames.forEach(nodeName => {
        const th = document.createElement('th');
        th.textContent = nodeName;
        tableHead.appendChild(th);
    });

    // Populate the table body with results for each iteration
    results.forEach((state, iteration) => {
        const row = document.createElement('tr');
        const iterationCell = document.createElement('td');
        iterationCell.textContent = iteration;
        row.appendChild(iterationCell);

        state.forEach(value => {
            const cell = document.createElement('td');
            cell.textContent = value.toFixed(5); // Show values with 5 decimal places
            row.appendChild(cell);
        });

        tableBody.appendChild(row);
    });

    // Step 1: Get the last row of the Kosko table (converged row)
    const koskoTable = document.getElementById('kosko-results-table');
    const koskoLastRow = koskoTable.querySelector('tbody tr:last-child');
    let koskoLastRowValues = [];

    if (koskoLastRow) {
        // Extract Kosko converged values from the last row
        koskoLastRow.querySelectorAll('td').forEach((cell, index) => {
            if (index > 0) { // Ignore the iteration number column
                koskoLastRowValues.push(parseFloat(cell.textContent));
            }
        });

        // Step 2: Add Kosko's converged row to Fixation table
        const koskoRow = document.createElement('tr');
        const koskoLabelCell = document.createElement('td');
        koskoLabelCell.textContent = 'Kosko Converged';  // Label for Kosko converged row
        koskoRow.appendChild(koskoLabelCell);

        koskoLastRowValues.forEach(value => {
            const cell = document.createElement('td');
            cell.textContent = value.toFixed(5);  // Keep the precision of 5 decimals
            koskoRow.appendChild(cell);
        });

        tableBody.appendChild(koskoRow);

        // Step 3: Calculate the difference between Fixation last row and Kosko converged row
        const fixationLastRow = results[results.length - 1];
        let differenceValues = [];  // Store the difference values for the bar chart

        const differenceRow = document.createElement('tr');
        const differenceLabelCell = document.createElement('td');
        differenceLabelCell.textContent = 'Difference (Fixation - Kosko)';  // Label for the difference row
        differenceRow.appendChild(differenceLabelCell);

        fixationLastRow.forEach((value, index) => {
            const difference = value - koskoLastRowValues[index];  // Calculate the difference
            differenceValues.push(difference);  // Store the difference for the bar chart

            const cell = document.createElement('td');
            cell.textContent = difference.toFixed(5);  // Show the difference with 5 decimals
            differenceRow.appendChild(cell);
        });

        // Step 4: Add the difference row to the Fixation table
        tableBody.appendChild(differenceRow);
        // Step 5: Render the difference row as a bar chart below the table
        renderDifferenceBarChart(differenceValues, nodeNames);
    } else {
        console.warn("Kosko results table doesn't have a converged row.");
    }
}

// Function to check if any node is tweaked
function isAnyNodeTweaked() {
    const nodeFields = document.querySelectorAll('.node-field');
    let isTweaked = false;

    nodeFields.forEach((nodeField) => {
        const tweakInput = nodeField.querySelector('input[placeholder="Tweak"]');
        const tweakValue = parseFloat(tweakInput.value);

        // If any tweak value is non-zero, we mark that a node is tweaked
        if (tweakValue !== 0) {
            isTweaked = true;
        }
    });

    return isTweaked;
}

// Function to render the difference row as a bar chart
function renderDifferenceBarChart(differenceValues, nodeNames) {
    // Check if any node is tweaked; if not, skip rendering the bar chart
    if (!isAnyNodeTweaked()) {
        console.log('No nodes are tweaked, skipping chart rendering.');
        return;
    }

    const ctx = document.getElementById('difference-bar-chart').getContext('2d');

    // Destroy the previous chart if it exists
    if (differenceBarChart) {
        differenceBarChart.destroy();
    }

    // Create gradient for the bars
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(255, 99, 132, 0.7)');
    gradient.addColorStop(1, 'rgba(54, 162, 235, 0.7)');

    // Create a new bar chart with a grid in the background
    differenceBarChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: nodeNames,  // Use the node names as labels for the chart
            datasets: [{
                label: 'Difference (Fixation - Kosko)',
                data: differenceValues,  // The difference values calculated
                backgroundColor: gradient,  // Apply gradient to fill
                borderColor: 'rgba(54, 162, 235, 1)',  // Dark blue border
                borderWidth: 2,
                hoverBackgroundColor: 'rgba(255, 206, 86, 0.8)',  // Hover effect color
                hoverBorderColor: 'rgba(255, 159, 64, 1)',  // Hover border color
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,  // Ensure y-axis starts at 0
                    grid: {
                        display: true,
                        color: 'rgba(201, 203, 207, 0.5)',  // Light grey gridlines for y-axis
                        lineWidth: 1,
                    }
                },
                x: {
                    grid: {
                        display: true,  // Display gridlines for x-axis as well
                        color: 'rgba(201, 203, 207, 0.5)',  // Light grey gridlines for x-axis
                        lineWidth: 1,
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Difference Between Fixation and Kosko Results',
                    font: {
                        size: 20,
                        weight: 'bold',
                    },
                    color: '#4B0082',  // Vibrant title color
                    padding: 20,
                },
                legend: {
                    display: true,
                    labels: {
                        color: '#333',  // Legend label color
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0,0,0,0.7)',  // Dark background for tooltips
                    titleColor: '#ffffff',  // White text for tooltip title
                    bodyColor: '#ffffff',  // White text for tooltip body
                }
            },
            animation: {
                duration: 1500,  // Slow animation for chart rendering
                easing: 'easeOutBounce'  // Bounce effect for visual interest
            },
            hover: {
                animationDuration: 1000,  // Smooth hover animation
            }
        }
    });
}


function checkAIReadiness() {
    fetch("/check_ai_readiness")
        .then(response => response.json())
        .then(data => {
            const aiStatusElement = document.getElementById("ai-status");
            const modelListContainer = document.getElementById("model-list-container");

            if (data.ollama_ready) {
                aiStatusElement.innerHTML = '<span style="color: green; font-weight: bold;">The system is AI-ready</span> for this app. <span style="color: black;"> If you do not see model of your choice, kindly manifest it through ollama.</span>';
                
                modelListContainer.innerHTML = "";  // Clear the container

                // Display the available models as buttons
                if (data.models.length > 0) {
                    data.models.forEach(model => {
                        const button = document.createElement("button");
                        button.className = "model-button";
                        button.innerText = model;
                        modelListContainer.appendChild(button);
                    });
                } else {
                    modelListContainer.innerHTML = "<p>No models available</p>";
                }
            } else {
                aiStatusElement.innerText = "AI system is not ready. Please ensure Ollama is running.";
                aiStatusElement.style.color = "red";  // Set text color to red
            }
        })
        .catch(error => {
            console.error("Error checking AI readiness:", error);
            const aiStatusElement = document.getElementById("ai-status");
            aiStatusElement.innerText = "Error checking AI readiness.";
            aiStatusElement.style.color = "red";  // Set text color to red in case of error
        });
}

// Get the overlay and the close button elements
var overlay = document.getElementById('summary-overlay');
var closeButton = document.getElementById('close-overlay-btn');

// Function to close the overlay
closeButton.addEventListener('click', function() {
    overlay.style.display = 'none';
});

function fetchAndShowSummary() {
    hideOverlay(); // Ensure the overlay is hidden initially

    // Fetch the summary from the backend
    fetch("/summarize_fcm", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            /* Your FCM data goes here */
        })
    })
    .then(response => response.json())
    .then(result => {
        // Replace **text** with <b>text</b> for bold formatting
        let summaryHtml = result.summary.replace(/\*\*(.*?)\*\*/g, '<b>$1</b>');  // Use regex to replace ** with <b>

        // Display the overlay with the formatted HTML summary
        document.getElementById('summary-text-overlay').innerHTML = summaryHtml;  // Inject formatted HTML
        document.getElementById('summary-overlay').style.display = 'block';
    })
    .catch(error => {
        console.error("Error:", error);
        // Optionally, show an error message in the overlay or alert
    });
}

// Close button logic
document.getElementById('close-overlay-btn').addEventListener('click', function() {
    document.getElementById('summary-overlay').style.display = 'none';
});

// You can trigger the fetchAndShowSummary() function when required
window.onload = 'hideOverlay'

// Function to hide the overlay
function hideOverlay() {
    const overlay = document.getElementById('overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

// Function to show the overlay with the fetched summary
function showOverlay(summaryHtml) {
    const overlay = document.getElementById('overlay');
    if (overlay) {
        // Set the inner HTML of the overlay to display the summary
        overlay.innerHTML = summaryHtml;
        overlay.style.display = 'block'; // Show the overlay
    }
}

// You can trigger the fetchAndShowSummary() function when required
window.onload = hideOverlay;  // Ensure overlay is hidden on page load

function sendFCMData() {
    // Set the selected model when the button is clicked
    document.getElementById('set-model-btn').addEventListener('click', function() {
        var selectedModel = document.getElementById('ollama-model-dropdown').value;
        fetch("/set_model", {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model_name: selectedModel })
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('model-status').innerText = "Model set to: " + data.model_name;
        })
        .catch(error => {
            console.error("Error setting model:", error);
        });
    });

    // Show loading overlay
    document.getElementById('loading-overlay').style.display = 'block';

    // Function to convert a table to a JSON object (all rows)
    function tableToJson(table) {
        var headers = [];
        var rows = [];
        // Get headers (column names)
        for (var i = 0; i < table.rows[0].cells.length; i++) {
            headers[i] = table.rows[0].cells[i].innerText.toLowerCase().replace(/\s/g, '_');
        }
        // Loop through the rows and create objects
        for (var i = 1; i < table.rows.length; i++) {
            var row = {};
            for (var j = 0; j < table.rows[i].cells.length; j++) {
                row[headers[j]] = table.rows[i].cells[j].innerText;
            }
            rows.push(row);
        }
        return rows;
    }

    // Function to extract node names from the first row of the weight matrix, skipping the first cell (which is 'iteration')
    function getNodeNamesFromWeightMatrix(table) {
        var nodeNames = [];
        for (var i = 1; i < table.rows[0].cells.length; i++) {  // Start at index 1 to skip the first cell
            nodeNames.push(table.rows[0].cells[i].innerText);  // Collect node names from the weight matrix
        }
        return nodeNames;
    }

    // Get node names from the first row of the weight matrix, skipping the first cell
    var nodeNames = getNodeNamesFromWeightMatrix(document.getElementById("weight-matrix-table"));

    // Convert the entire node stats table to JSON
    var nodeStatsJson = tableToJson(document.getElementById("node-stats-table"));
    
    // Convert the entire Kosko results table to JSON
    var koskoResultsJson = tableToJson(document.getElementById("kosko-results-table"));
    
    // Convert the entire fixation results table to JSON
    var fixationResultsJson = tableToJson(document.getElementById("fixation-results-table"));

    // Create a data object to send to the Flask server
    var data = {
        node_stats: nodeStatsJson,  // The whole node stats table
        kosko_results: koskoResultsJson,  // The whole Kosko results table
        fixation_results: fixationResultsJson  // The whole fixation results table
    };

    // Send the data using Fetch API to Flask
    fetch("/summarize_fcm", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        // Handle the response and display the summary
        document.getElementById("summary-text-overlay").innerHTML = result.summary;  // Assuming summary is HTML

        // Show the overlay with the summary
        document.getElementById("summary-overlay").style.display = "flex";
    })
    .then(data => {
        // Hide the loading overlay when the response is received
        document.getElementById('loading-overlay').style.display = 'none';
    })
    .catch(error => {
        console.error("Error:", error);
        // Hide the loading overlay in case of error
        document.getElementById('loading-overlay').style.display = 'none';
    });
}

function fetchAvailableModels() {
    fetch("/get_available_models")
        .then(response => response.json())
        .then(data => {
            var modelDropdown = document.getElementById('ollama-model-dropdown');
            modelDropdown.innerHTML = ''; // Clear existing options
            data.models.forEach(model => {
                var option = document.createElement('option');
                option.value = model;
                option.text = model;
                modelDropdown.appendChild(option);
            });
        })
        .catch(error => {
            console.error("Error fetching models:", error);
        });
}

// Set the selected model when the button is clicked
document.getElementById('set-model-btn').addEventListener('click', function() {
    var selectedModel = document.getElementById('ollama-model-dropdown').value;
    fetch("/set_model", {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ model_name: selectedModel })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('model-status').innerText = "Model set to: " + data.model_name;
    })
    .catch(error => {
        console.error("Error setting model:", error);
    });
});

// Call the function to fetch models when the page loads
window.onload = fetchAvailableModels();