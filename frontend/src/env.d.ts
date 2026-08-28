// Minimal typing for process.env in the browser bundle.
// This avoids pulling in full Node.js typings just to use REACT_APP_* vars.

declare const process: {
  env: {
    REACT_APP_API_URL?: string;
    [key: string]: string | undefined;
  };
};


