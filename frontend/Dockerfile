# Build: TanStack Start in SPA mode writes static files to dist/client, with
# dist/client/_shell.html for every path that is not a file.
FROM node:25-alpine AS build

WORKDIR /app

# Install and build as `node`, so no `chown -R` layer duplicates node_modules.
RUN chown node:node /app
USER node

COPY --chown=node:node package.json package-lock.json ./
RUN npm ci

COPY --chown=node:node . .

# VITE_* values are inlined into the bundle, so they are build arguments.
ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

RUN npm run build

# Serve: nginx as a non-root user on 8080. Nothing runs Node at this stage.
FROM nginxinc/nginx-unprivileged:stable-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist/client /usr/share/nginx/html

EXPOSE 8080
