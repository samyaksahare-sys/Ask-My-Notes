# ---------- stage 1: build the Vite bundle ----------
FROM node:22-alpine AS build

WORKDIR /app

# Lockfile-only install first so this layer caches across source edits.
COPY package.json package-lock.json ./
RUN npm ci

# Vite inlines import.meta.env at BUILD time. This value is therefore baked
# into the JS bundle and CANNOT be changed by a runtime environment variable.
# It must be the URL the BROWSER can reach -- the browser runs on the host, so
# it is the published backend port, never http://backend:8000.
ARG VITE_API_BASE=http://localhost:8000
ENV VITE_API_BASE=$VITE_API_BASE

COPY . .
RUN npm run build

# ---------- stage 2: serve the static files ----------
FROM nginx:alpine AS serve

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
# nginx:alpine already runs nginx in the foreground as its CMD.
