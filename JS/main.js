let pixelArrayZ = [
    // 10x010 front
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
];

/**
 * Label the directions in the 3D viewport scene
 */
let _3DFaceLabelsMapStrZ = new Map();

_3DFaceLabelsMapStrZ.set('front', 0);
_3DFaceLabelsMapStrZ.set('back', 1);
_3DFaceLabelsMapStrZ.set('left', 2);
_3DFaceLabelsMapStrZ.set('right',3);
_3DFaceLabelsMapStrZ.set('bottom', 4);
_3DFaceLabelsMapStrZ.set('top', 5);

/**
Simulate the allotted viewable angles and faces for the viewport
*/
const controllerViewportAnglesRangeObjZ = {
    min: 0,
    max: 5,
}

/**
Simulate the number of pixelArrayZ objects (ie. animation frames) to be presented onscreen
*/
const animationSequenceRangeObjZ = {
    begin: 0,
    end: 1,
    step: 0.001,
}

/**
	@brief Performs linear interpolation (LERP) between two values.
	@details Calculates a value linearly interpolated between a start and end value,
			 based on a given factor that indicates the relative position between the two.
	@param start The start value for interpolation.
	@param end The end value for interpolation.
	@param factor Range between [0.000,1.000] indicating the position between the start and end values.
	@return The interpolated value. */
	function LERP(start, end, factor) {
		return (1 - factor) * start + factor * end;
	}

/*
Scenarios

Depending on the current viewport perpective, 
the animation consists of 1000 frame 3D flipbook, 
with each frame rendered, intended for rasterization 
display in pixelArrayZ!



Theory

There are a few strategies available to determine the final rasterization image
to be displayed to the screen:

1. Calculate IK/FK resolves for the actual 3D mesh within the gme environment, 
in realtime, while accounting for player / controller input (requires additional gpu/cpu)

2. Cache all animation perspectives for the 3D mesh, 
for later playback; then present the next frame based on controller-input 
and the environment (requires additional logic (cpu) + memory)

3. Employ a combination Convolutional Neural Network (CNN)
plus either a Long Short-Term Memory (LSTM) model  
or Transformer model to capture spatial- and temporal- 
information, respectively, from the game environment 
and player input-controller.

*/

// Function to simulate the gamePad
function getGamepads() {
    const gamepads = [
        {
            id: "Xbox Controller (STANDARD GAMEPAD Vendor: 045e Product: 02fd)",
            index: 0,
            connected: true,
            buttons: [
                { pressed: false }, // A
                { pressed: false }, // B
                { pressed: false }, // X
                { pressed: false }, // Y
                { pressed: false }, // Left Bumper
                { pressed: false }, // Right Bumper
                { pressed: false, value: 0 }, // Left Trigger
                { pressed: false, value: 0 }, // Right Trigger
                { pressed: false }, // Back
                { pressed: false }, // Start
                { pressed: false }, // Left Stick
                { pressed: false }, // Right Stick
                { pressed: false }, // D-pad Up
                { pressed: false }, // D-pad Down
                { pressed: false }, // D-pad Left
                { pressed: false }, // D-pad Right
                { pressed: false }, // Xbox Button
            ],
            axes: [0, 0, 0, 0], // [leftStickX, leftStickY, rightStickX, rightStickY]
        },
        {
            id: "PlayStation 5 Controller (STANDARD GAMEPAD Vendor: 054c Product: 0ce6)",
            index: 1,
            connected: true,
            buttons: [
                { pressed: false }, // Cross
                { pressed: false }, // Circle
                { pressed: false }, // Square
                { pressed: false }, // Triangle
                { pressed: false }, // L1
                { pressed: false }, // R1
                { pressed: false, value: 0 }, // L2
                { pressed: false, value: 0 }, // R2
                { pressed: false }, // Share
                { pressed: false }, // Options
                { pressed: false }, // L3
                { pressed: false }, // R3
                { pressed: false }, // D-pad Up
                { pressed: false }, // D-pad Down
                { pressed: false }, // D-pad Left
                { pressed: false }, // D-pad Right
                { pressed: false }, // PS Button
                { pressed: false }, // Touchpad
            ],
            axes: [0, 0, 0, 0], // [leftStickX, leftStickY, rightStickX, rightStickY]
        }
    ];
    return gamepads;
}

// Function to simulate button press
function pressButton(gamepadIndex, buttonIndex) {
    const gamepads = getGamepads();
    if (gamepads[gamepadIndex] && gamepads[gamepadIndex].buttons[buttonIndex]) {
        gamepads[gamepadIndex].buttons[buttonIndex].pressed = true;
    }
}

// Function to simulate button release
function releaseButton(gamepadIndex, buttonIndex) {
    const gamepads = getGamepads();
    if (gamepads[gamepadIndex] && gamepads[gamepadIndex].buttons[buttonIndex]) {
        gamepads[gamepadIndex].buttons[buttonIndex].pressed = false;
    }
}

// Function to simulate analog trigger press
function pressTrigger(gamepadIndex, triggerIndex, value) {
    const gamepads = getGamepads();
    if (gamepads[gamepadIndex] && gamepads[gamepadIndex].buttons[triggerIndex]) {
        gamepads[gamepadIndex].buttons[triggerIndex].pressed = value > 0;
        gamepads[gamepadIndex].buttons[triggerIndex].value = Math.max(0, Math.min(1, value));
    }
}

// Function to simulate stick movement
function moveStick(gamepadIndex, axisIndex, value) {
    const gamepads = getGamepads();
    if (gamepads[gamepadIndex] && gamepads[gamepadIndex].axes[axisIndex] !== undefined) {
        gamepads[gamepadIndex].axes[axisIndex] = Math.max(-1, Math.min(1, value));
    }
}

// Function to handle gamepad connection
function handleGamepadConnected(event) {
    console.log("Gamepad connected:");
    console.log(event.gamepad);
} // end handleGamepadConnected()

// Function to handle gamepad disconnection
function handleGamepadDisconnected(event) {
    console.log("Gamepad disconnected:");
    console.log(event.gamepad);
} // end handleGamepadDisconnected()

// Function to process gamepad input
function processGamepadInput() {
    const gamepads = /* navigator. */getGamepads();
    if (!gamepads) {
        return;
    }

    const gamepad = gamepads[0]; // Assuming we're using the first connected gamepad
    if (!gamepad) {
        return;
    }

    // Process button inputs
    for (let i = 0; i < gamepad.buttons.length; i++) {
        const button = gamepad.buttons[i];
        if (button.pressed) {
            console.log(`Button ${i} pressed`);
            // Add your button-specific logic here
        }
    }

    // Process analog stick inputs
    const leftStickX = gamepad.axes[0];
    const leftStickY = gamepad.axes[1];
    const rightStickX = gamepad.axes[2];
    const rightStickY = gamepad.axes[3];

    // Example: Check if left stick is pushed significantly to the right
    if (leftStickX > 0.5) {
        console.log("Left stick pushed right");
    }

    // Add more analog stick processing logic here

    // Request the next animation frame
    requestAnimationFrame(processGamepadInput);
} // end processGamepadInput()

// Set up event listeners
window.addEventListener("gamepadconnected", handleGamepadConnected);
window.addEventListener("gamepaddisconnected", handleGamepadDisconnected);

// Start processing gamepad input
requestAnimationFrame(processGamepadInput);

// Example usage
console.log(getGamepads()); // Initial state

const Platform = {
    Console: {
        XBOX: {
            index: 0,
            Button: {
                A: 0,
                B: 1,
                X: 2,
                Y: 3,
                LB: 4,
                RB: 5,
                BACK: 8,
                START: 9,
                LEFT_STICK: 10,
                RIGHT_STICK: 11,
                DPAD_UP: 12,
                DPAD_DOWN: 13,
                DPAD_LEFT: 14,
                DPAD_RIGHT: 15,
                XBOX: 16,
            },
            Trigger: {
                LT: 6,
                RT: 7,
            },
            Axis: {
                LEFT_STICK_X: 0,
                LEFT_STICK_Y: 1,
                RIGHT_STICK_X: 2,
                RIGHT_STICK_Y: 3,
            },
        },
        PS5: {
            index: 1,
            Button: {
                CROSS: 0,
                CIRCLE: 1,
                SQUARE: 2,
                TRIANGLE: 3,
                L1: 4,
                R1: 5,
                SHARE: 8,
                OPTIONS: 9,
                L3: 10,
                R3: 11,
                DPAD_UP: 12,
                DPAD_DOWN: 13,
                DPAD_LEFT: 14,
                DPAD_RIGHT: 15,
                PS: 16,
                TOUCHPAD: 17,
            },
            Trigger: {
                L2: 6,
                R2: 7,
            },
            Axis: {
                LEFT_STICK_X: 0,
                LEFT_STICK_Y: 1,
                RIGHT_STICK_X: 2,
                RIGHT_STICK_Y: 3,
            },
        },
    },
    PC: {
        index: 2,
        Key: {
            W: 'w',
            A: 'a',
            S: 's',
            D: 'd',
            SPACE: ' ',
            SHIFT: 'Shift',
            CTRL: 'Control',
            ALT: 'Alt',
            ENTER: 'Enter',
            ESCAPE: 'Escape',
            ARROW_UP: 'Up',
            ARROW_DOWN: 'Down',
            ARROW_LEFT: 'Left',
            ARROW_RIGHT: 'Right',
            // Add keys as needed
        },
        Mouse: {
            LEFT: 0,
            MIDDLE: 1,
            RIGHT: 2,
        },
    },
}; // end Platform

pressButton(Platform.Console.XBOX, Platform.Console.XBOX.Button.A); // Press A on Xbox controller
pressTrigger(Platform.Console.PS5, Platform.Console.PS5.Trigger.R2, 0.5); // Half-press R2 on PS5 controller
moveStick(Platform.Console.XBOX, Platform.Console.XBOX.Axis.LEFT_STICK_X, 0.7); // Move left stick right on Xbox controller