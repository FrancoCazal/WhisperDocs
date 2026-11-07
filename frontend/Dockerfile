FROM node:22-alpine

WORKDIR /app

# Install and build as `node`, so no `chown -R` layer duplicates node_modules.
RUN chown node:node /app
USER node

COPY --chown=node:node package.json package-lock.json ./
RUN npm ci

COPY --chown=node:node . .

# VITE_* values are inlined into the bundle, so they are build arguments.
ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL \
    BUILD_TARGET=node

# A production build, served by `vite preview`: the app server-renders, so it
# needs dist/server/server.js and cannot be served as static files. Preview
# loads vite.config.ts and its plugins at startup, so node_modules stays.
RUN npm run build

EXPOSE 8080

CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0", "--port", "8080", "--strictPort"]
